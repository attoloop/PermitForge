"""HF generation with owned hybrid state across tool turns; no model or sampler copy."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import time


@dataclass
class Entry:
    identity: int
    tokens: list[int]
    consumed: int
    cache: object | None


def cache_tensors(cache) -> dict:
    result = {}
    for i,layer in enumerate(cache.layers):
        for name in ('keys','values'):
            if (value := getattr(layer,name,None)) is not None:
                result[f'{i}/{name}'] = value
        for name in ('conv_states','recurrent_states'):
            for j,value in getattr(layer,name,{}).items():
                if value is not None:
                    result[f'{i}/{name}/{j}'] = value
    return result


def merge_caches(entries: list[Entry], config):
    import torch
    import torch.nn.functional as F
    from transformers import DynamicCache
    lengths = [e.consumed for e in entries]
    maximum = max(lengths)
    merged = DynamicCache(config=config)
    if maximum == 0:
        return merged
    if min(lengths) == 0:
        raise ValueError('Initialize empty caches separately before batching decode')
    for index,target in enumerate(merged.layers):
        sources = [e.cache.layers[index] for e in entries]
        if hasattr(target,'keys'):
            kv = [torch.cat([F.pad(getattr(s,name),(0,0,maximum-n,0))
                              for s,n in zip(sources,lengths,strict=True)],dim=0)
                  for name in ('keys','values')]
            target.update(*kv)
        if hasattr(target,'conv_states'):
            for state_index in sources[0].conv_states:
                target.update_conv_state(torch.cat([s.conv_states[state_index] for s in sources]),
                    state_idx=state_index,conv_kernel_size=sources[0].conv_kernel_size[state_index])
                target.update_recurrent_state(torch.cat([s.recurrent_states[state_index] for s in sources]),
                                               state_idx=state_index)
    return merged


def split_cache(cache, lengths: list[int], config) -> list:
    from transformers import DynamicCache
    result = []
    for row,length in enumerate(lengths):
        single = DynamicCache(config=config)
        for target,source in zip(single.layers,cache.layers,strict=True):
            if hasattr(target,'keys'):
                target.update(source.keys[row:row+1,:,-length:].clone(),
                              source.values[row:row+1,:,-length:].clone())
            if hasattr(target,'conv_states'):
                for j in source.conv_states:
                    target.update_conv_state(source.conv_states[j][row:row+1].clone(),state_idx=j,
                                             conv_kernel_size=source.conv_kernel_size[j])
                    target.update_recurrent_state(source.recurrent_states[j][row:row+1].clone(),state_idx=j)
        result.append(single)
    return result


class HybridGenerator:
    def __init__(self, model, *, pad_token_id: int, microbatch: int = 1, persistent: bool = False):
        if microbatch < 1:
            raise ValueError('Positive generation microbatch required')
        self.model = model
        self.pad = pad_token_id
        self.microbatch = microbatch
        self.persistent = persistent
        self.entries: list[Entry] = []
        self.next_identity = 0
        self.stats = dict(reused_tokens=0,prefill_tokens=0,invalidations=[],turns=[])

    def invalidate(self, reason: str) -> None:
        self.entries.clear()
        self.stats['invalidations'].append(reason)

    def _claim(self, prompts: list[list[int]]) -> list[Entry]:
        available = list(self.entries) if self.persistent else []
        claimed = []
        for prompt in prompts:
            if not prompt:
                raise ValueError('Empty generation prompt')
            matches = [e for e in available if len(e.tokens) < len(prompt) and
                       prompt[:len(e.tokens)] == e.tokens]
            if matches:
                entry = max(matches,key=lambda e:len(e.tokens))
                available.remove(entry)
                # An entry is transferred once, never shared between sibling rollouts.
                self.stats['reused_tokens'] += entry.consumed
                entry.tokens = list(prompt)
            else:
                entry = Entry(self.next_identity,list(prompt),0,None)
                self.next_identity += 1
            claimed.append(entry)
        self.entries = claimed
        return claimed

    def _sync(self):
        import torch
        if self.model.device.type == 'cuda':
            torch.cuda.synchronize()

    def _inputs(self, entries: list[Entry], *, suffix_only: bool):
        import torch
        prefix = max(e.consumed for e in entries)
        suffix = len(entries[0].tokens)-entries[0].consumed
        if any(len(e.tokens)-e.consumed != suffix for e in entries):
            raise ValueError('A continuing recurrent state must never receive suffix padding')
        ids,masks,positions = [],[],[]
        for entry in entries:
            n = prefix-entry.consumed
            ids.append(entry.tokens[entry.consumed:] if suffix_only else [self.pad]*n+entry.tokens)
            masks.append([0]*n+[1]*len(entry.tokens))
            positions.append(list(range(entry.consumed,len(entry.tokens))))
        device = self.model.device
        return dict(input_ids=torch.tensor(ids,device=device),attention_mask=torch.tensor(masks,device=device),
                    **({'position_ids':torch.tensor(positions,device=device)} if suffix_only else {}))

    def _forward(self, entries: list[Entry]):
        import torch
        cache = merge_caches(entries,self.model.config)
        with torch.no_grad():
            result = self.model(**self._inputs(entries,suffix_only=True),past_key_values=cache,
                                use_cache=True,logits_to_keep=1)
        lengths = [len(e.tokens) for e in entries]
        caches = split_cache(result.past_key_values,lengths,self.model.config)
        for entry,state in zip(entries,caches,strict=True):
            self.stats['prefill_tokens'] += len(entry.tokens)-entry.consumed
            entry.consumed = len(entry.tokens)
            entry.cache = state
        return list(result.logits[:,-1].detach().unbind(0))

    def _prefill(self, entries: list[Entry], *, leave_last: bool):
        groups = defaultdict(list)
        for index,entry in enumerate(entries):
            target = len(entry.tokens)-int(leave_last)
            if target <= entry.consumed:
                continue
            # No left padding is ever applied to the suffix of a nonzero GDN state.
            groups[(entry.consumed == 0,target-entry.consumed)].append((index,entry))
        logits = {}
        for members in groups.values():
            for start in range(0,len(members),self.microbatch):
                chunk = members[start:start+self.microbatch]
                batch = [e for _,e in chunk]
                last = [e.tokens.pop() for e in batch] if leave_last else []
                try:
                    values = self._forward(batch)
                    logits.update({i:v for (i,_),v in zip(chunk,values,strict=True)})
                finally:
                    if leave_last:
                        for e,token in zip(batch,last,strict=True):
                            e.tokens.append(token)
        return logits

    def teacher_force(self, prompts: list[list[int]]):
        entries = self._claim(prompts)
        values = self._prefill(entries,leave_last=False)
        return [values[i] for i in range(len(entries))]

    def assert_state(self, index: int, expected, **kwargs) -> None:
        import torch
        actual,reference = cache_tensors(self.entries[index].cache),cache_tensors(expected)
        assert actual.keys() == reference.keys()
        for key in actual:
            torch.testing.assert_close(actual[key],reference[key],**kwargs)

    def generate(self, prompts: list[list[int]], generation_config, *, forced_tokens=None) -> list[list[int]]:
        import torch
        from transformers import LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
        cfg = deepcopy(generation_config)
        if (cfg.num_beams or 1) != 1 or (cfg.num_return_sequences or 1) != 1 or not cfg.max_new_tokens:
            raise ValueError('Qualified HF generation requires one beam/return and max_new_tokens')
        eos = cfg.eos_token_id
        eos = [] if eos is None else eos if isinstance(eos,list) else [eos]
        class StopTogether(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):
                stop = any(bool((input_ids[:,-1] == token).any()) for token in eos)
                return torch.full((input_ids.shape[0],),stop,device=input_ids.device,dtype=torch.bool)
        self._sync()
        started = time.monotonic()
        reused = self.stats['reused_tokens']
        entries = self._claim(prompts)
        output = [[] for _ in prompts]
        self._prefill(entries,leave_last=True)
        self._sync()
        prefill_seconds = time.monotonic()-started
        pending = list(range(len(entries)))
        while pending:
            # Initial prefill leaves exactly one token; each HF return also trails one token.
            group = pending[:self.microbatch]
            del pending[:self.microbatch]
            batch = [entries[i] for i in group]
            # Different remaining caps cannot be padded or silently shortened.
            remaining = min(cfg.max_new_tokens-len(output[i]) for i in group)
            if min(e.consumed for e in batch) == 0 and max(e.consumed for e in batch) > 0:
                pending[0:0] = group[1:]
                group,batch = group[:1],batch[:1]
            inputs = self._inputs(batch,suffix_only=False)
            width = inputs['input_ids'].shape[1]
            cache = merge_caches(batch,self.model.config)
            processors = LogitsProcessorList()
            if forced_tokens is not None:
                offsets = [len(output[i]) for i in group]
                def force(ids,scores,group=group,offsets=offsets,width=width):
                    scores.fill_(-torch.inf)
                    for row,i in enumerate(group):
                        scores[row,forced_tokens[i][offsets[row]+ids.shape[1]-width]] = 0
                    return scores
                processors.append(force)
            with torch.no_grad():
                result = self.model.generate(**inputs,past_key_values=cache,generation_config=cfg,
                    max_new_tokens=remaining,return_dict_in_generate=True,use_cache=True,logits_to_keep=1,
                    stopping_criteria=StoppingCriteriaList([StopTogether()]),logits_processor=processors)
            new = result.sequences[:,width:].tolist()
            lengths = [len(e.tokens)+len(tokens)-1 for e,tokens in zip(batch,new,strict=True)]
            states = split_cache(result.past_key_values,lengths,self.model.config)
            for i,e,tokens,state,length in zip(group,batch,new,states,lengths,strict=True):
                if not tokens:
                    raise RuntimeError('HF generation returned an empty segment')
                e.tokens.extend(tokens)
                e.consumed,e.cache = length,state
                output[i].extend(tokens)
                if tokens[-1] not in eos and len(output[i]) < cfg.max_new_tokens:
                    pending.append(i)
        self._sync()
        elapsed = time.monotonic()-started
        self.stats['turns'].append(dict(seconds=elapsed,prefill_seconds=prefill_seconds,
            decode_seconds=elapsed-prefill_seconds,batch_size=len(prompts),microbatch=self.microbatch,
            input_tokens=sum(map(len,prompts)),output_tokens=sum(map(len,output)),
            reused_tokens=self.stats['reused_tokens']-reused))
        if not self.persistent:
            self.entries.clear()
        return output


def cached_trainer_class(base, *, microbatch: int, persistent: bool):
    import torch
    from transformers import TrainerCallback
    from trl.models import unwrap_model_for_generation
    class CachedTrainer(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.hybrid_generator = None
            self.generation_timings = []
            trainer = self
            class Invalidate(TrainerCallback):
                def on_step_end(self,*args,**kwargs):
                    if trainer.hybrid_generator is not None:
                        trainer.hybrid_generator.invalidate('optimizer_step')
            self.add_callback(Invalidate())

        def _generate_single_turn(self,prompt_ids,images,multimodal_fields,has_tool_images=False):
            if images or multimodal_fields or has_tool_images or self.use_vllm:
                raise ValueError('Qualified generation is HF text-only')
            settings = getattr(self,'stack_settings',dict(microbatch=microbatch,persistent=persistent))
            if not settings['persistent']:
                started = time.monotonic()
                result = super()._generate_single_turn(prompt_ids,images,multimodal_fields,has_tool_images)
                if self.accelerator.device.type == 'cuda':
                    torch.cuda.synchronize()
                self.generation_timings.append(dict(seconds=time.monotonic()-started,
                    batch_size=len(prompt_ids),microbatch=1,persistent=False,
                    input_tokens=sum(map(len,prompt_ids)),output_tokens=sum(map(len,result[0]))))
                return result
            with (unwrap_model_for_generation(self.model_wrapped,self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                    generation_kwargs=self.generation_kwargs) as model,
                  torch.autocast(device_type=self.accelerator.device.type,dtype=torch.bfloat16,
                                 enabled=self.accelerator.device.type == 'cuda')):
                if self.hybrid_generator is None or self.hybrid_generator.model is not model:
                    self.hybrid_generator = HybridGenerator(model,pad_token_id=self._tokenizer.pad_token_id,
                        microbatch=settings['microbatch'],persistent=settings['persistent'])
                tokens = self.hybrid_generator.generate(prompt_ids,self.generation_config)
                self.generation_timings.append(self.hybrid_generator.stats['turns'][-1])
            return tokens,None

        def _generate(self,prompts):
            if self.hybrid_generator is not None:
                self.hybrid_generator.invalidate('rollout_group_start')
            try:
                return super()._generate(prompts)
            finally:
                if self.hybrid_generator is not None:
                    self.hybrid_generator.invalidate('rollout_group_end')
                torch.cuda.empty_cache()
    return CachedTrainer
