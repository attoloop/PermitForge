"""Pure configuration and planning arithmetic: python -m train.cloud.run --plan."""
from dataclasses import asdict, dataclass
import math

STAGES = ('fields', 'base-eval', 'smoke', 'grpo', 'eval', 'export')
FIELD_CPU_SECONDS = 480       # measured: five smoke tasks in about ten minutes on four workers
EVAL_EPISODE_SECONDS = 816    # ceil(mean complete base episodes), including native grading; test_eval_rules.py
GRPO_STEP_SECONDS = 1700      # estimate for six rollouts: 289 s measured at 8 turns with four; eval-length rollouts run about four times longer
GRPO_STEP_CAP_SECONDS = 3400  # cap per step: twice the estimate
PRICES = {'hf-jobs': 5.00}


@dataclass(frozen=True)
class CloudConfig:
    hourly_usd: float = 5.00
    flavor: str = 'h200'
    precision: str = 'bf16'
    recipe: str = 'smoke'
    trackio_space: str | None = None
    trackio_bucket: str | None = None
    model: str = 'Qwen/Qwen3.8-27B'
    revision: str = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
    field_jobs: int = 4
    use_vllm: bool = False
    max_vram_bytes: int = 135_000_000_000
    sft_steps: int = 100
    grpo_steps: int = 100
    max_length: int = 45056
    max_completion_length: int = 40960
    max_prompt_length: int = 4096
    train_turns: int = 40
    num_generations: int = 6
    accum: int = 6
    lora_rank: int = 16
    lora_alpha: int = 32
    seed: int = 0
    eval_repeats: int = 3
    eval_turns: int = 40
    eval_max_tokens: int = 32768
    eval_wall_seconds: int = 1800
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.hourly_usd) or self.hourly_usd <= 0:
            raise ValueError('hourly_usd must be finite and positive')
        if self.model != 'Qwen/Qwen3.8-27B' or self.revision != '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0':
            raise ValueError('Cloud recipe requires the WP2.3a model revision')
        if not 1 <= self.field_jobs <= 4 or self.use_vllm or self.enable_thinking:
            raise ValueError('Use at most four native workers, HF generation and thinking off')
        if self.flavor != 'h200' or self.precision not in ('bf16', 'nf4'):
            raise ValueError('Use H200 with bf16 LoRA or explicit NF4')
        if self.recipe not in ('smoke', 'grpo'):
            raise ValueError('Unknown recipe')
        if self.trackio_bucket and not self.trackio_space:
            raise ValueError('Trackio bucket requires an explicitly named Space')
        for destination in (self.trackio_space, self.trackio_bucket):
            if destination and (destination.split('/')[0].lower() != 'attoloop' or destination.count('/') != 1):
                raise ValueError('Name Trackio resources under Attoloop')
        if not 0 < self.max_vram_bytes <= 135_000_000_000:
            raise ValueError('VRAM abort threshold cannot exceed 135 decimal GB')
        for key in ('sft_steps', 'grpo_steps', 'max_length', 'max_completion_length',
                    'max_prompt_length', 'train_turns', 'num_generations', 'accum'):
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise ValueError(key + ' must be a positive integer')
        if self.num_generations < 2 or self.accum % self.num_generations:
            raise ValueError('accum must be divisible by num_generations >= 2')
        if self.train_turns != self.eval_turns:
            raise ValueError('Training follows the evaluation: train_turns must equal eval_turns')
        if self.max_prompt_length + self.max_completion_length > self.max_length:
            raise ValueError('Prompt plus completion must fit the training context')
        if ((self.eval_repeats not in ((1,3) if self.recipe == 'smoke' else (3,))) or
                (self.eval_turns, self.eval_max_tokens, self.eval_wall_seconds) != (40,32768,1800)):
            raise ValueError('Frozen cloud evaluation uses 10 tasks x 3, 40 turns, 32768 tokens, 1800 s')

    def record(self) -> dict:
        return asdict(self)


def remaining_seconds(wall: float, usd: float, hourly: float, attempts: list[dict]) -> float:
    if any(not math.isfinite(x) or x <= 0 for x in (wall, usd, hourly)):
        raise ValueError('Caps and hourly rate must be finite and positive')
    charged = sum(a.get('charged_seconds', a.get('reserved_seconds', 0)) for a in attempts)
    return max(0., min(wall, usd * 3600 / hourly) - charged)


def stage_plan(config: CloudConfig, train_count: int = 30,
               wall_caps: dict | None = None, usd_caps: dict | None = None,
               *, stop_after_step: int | None = None) -> list[dict]:
    # Planning inputs (README, Plans): fields, eval episodes and the kernel check were measured on the H200
    # smoke Attoloop/6ad478f2c662881452271a2c; the eval-rules GRPO step is an estimate until a smoke measures it.
    steps = stop_after_step or config.grpo_steps
    estimates = {'fields': (train_count+10)*FIELD_CPU_SECONDS/config.field_jobs, 'base-eval': 30*EVAL_EPISODE_SECONDS,
        'smoke': 5*FIELD_CPU_SECONDS/config.field_jobs+3*EVAL_EPISODE_SECONDS+2*GRPO_STEP_SECONDS+180+100,
        'grpo': steps*GRPO_STEP_SECONDS, 'eval':30*EVAL_EPISODE_SECONDS, 'export':300}
    walls = {'fields':2*(train_count+10)*FIELD_CPU_SECONDS//config.field_jobs, 'base-eval':30*1200, 'smoke':10200,
             'grpo':config.grpo_steps*GRPO_STEP_CAP_SECONDS+1800, 'eval':30*1200, 'export':900}
    stages = ['smoke'] if config.recipe == 'smoke' else ['fields','base-eval','grpo','eval','export']
    if stop_after_step and stop_after_step < config.grpo_steps:
        stages = stages[:3]
    rows = []
    for stage in stages:
        wall = (wall_caps or {}).get(stage, walls[stage])
        usd = (usd_caps or {}).get(stage, wall*config.hourly_usd/3600)
        effective = remaining_seconds(wall, usd, config.hourly_usd, [])
        if stage == 'grpo' and stop_after_step:
            effective = min(effective, steps*GRPO_STEP_CAP_SECONDS+1800)
        if stage == 'smoke' and (wall > 10200 or usd > 15):
            raise ValueError('Smoke caps may only be tightened (10200 seconds, $15)')
        rows.append(dict(stage=stage, estimated_seconds=estimates[stage],
            estimated_usd=estimates[stage]*config.hourly_usd/3600,
            wall_seconds=wall, usd_cap=usd,
            effective_seconds=effective,
            **({'episodes':30} if stage in ('base-eval', 'eval') else {}),
            **({'optimizer_steps_this_invocation':steps} if stage == 'grpo' else {})))
    return rows
