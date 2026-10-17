"""Pure configuration and planning arithmetic: python -m train.cloud.run --plan."""
from dataclasses import asdict, dataclass
import math

STAGES = ('fields', 'base-eval', 'smoke', 'sft', 'grpo', 'eval', 'export')
WALL_HOURS = dict(zip(STAGES, (8, 16, 1, 2, 8, 16, 1)))
PRICES = {'hf-jobs': 2.75, 'jarvislabs': 2.19, 'runpod': 2.49}


@dataclass(frozen=True)
class CloudConfig:
    hourly_usd: float
    model: str = 'Qwen/Qwen3.8-27B'
    revision: str = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
    field_jobs: int = 4
    use_vllm: bool = False
    max_vram_bytes: int = 90_000_000_000
    sft_steps: int = 100
    grpo_steps: int = 100
    max_length: int = 16384
    max_completion_length: int = 8192
    max_prompt_length: int = 8192
    train_turns: int = 8
    num_generations: int = 4
    accum: int = 4
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
        if not 0 < self.max_vram_bytes <= 90_000_000_000:
            raise ValueError('VRAM abort threshold cannot exceed 90 decimal GB')
        for key in ('sft_steps', 'grpo_steps', 'max_length', 'max_completion_length',
                    'max_prompt_length', 'train_turns', 'num_generations', 'accum'):
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise ValueError(key + ' must be a positive integer')
        if self.num_generations < 2 or self.accum % self.num_generations:
            raise ValueError('accum must be divisible by num_generations >= 2')
        if self.max_prompt_length + self.max_completion_length > self.max_length:
            raise ValueError('Prompt plus completion must fit the training context')
        if (self.eval_repeats, self.eval_turns, self.eval_max_tokens, self.eval_wall_seconds) != (3, 40, 32768, 1800):
            raise ValueError('Frozen cloud evaluation uses 10 tasks x 3, 40 turns, 32768 tokens, 1800 s')

    def record(self) -> dict:
        return asdict(self)


def remaining_seconds(wall: float, usd: float, hourly: float, attempts: list[dict]) -> float:
    if any(not math.isfinite(x) or x <= 0 for x in (wall, usd, hourly)):
        raise ValueError('Caps and hourly rate must be finite and positive')
    charged = sum(a.get('charged_seconds', a.get('reserved_seconds', 0)) for a in attempts)
    return max(0., min(wall, usd * 3600 / hourly) - charged)


def stage_plan(config: CloudConfig, train_count: int = 230,
               wall_caps: dict | None = None, usd_caps: dict | None = None) -> list[dict]:
    # Explicit unmeasured priors. Native work includes eval fields. No speed claim.
    estimates = dict(zip(STAGES, ((train_count+10)*300/config.field_jobs, 30*180,
        1800, config.sft_steps*20, config.grpo_steps*180, 30*180, 900)))
    rows = []
    for stage in STAGES:
        wall = (wall_caps or {}).get(stage, WALL_HOURS[stage]*3600)
        usd = (usd_caps or {}).get(stage, wall*config.hourly_usd/3600)
        rows.append(dict(stage=stage, estimated_seconds=estimates[stage],
            estimated_usd=estimates[stage]*config.hourly_usd/3600,
            wall_seconds=wall, usd_cap=usd,
            effective_seconds=remaining_seconds(wall, usd, config.hourly_usd, []),
            **({'episodes':30} if stage in ('base-eval', 'eval') else {})))
    return rows
