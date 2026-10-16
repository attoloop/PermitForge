"""Episode geometry and wind counts from the task's AERMET surface file."""
from bisect import bisect_right
from functools import lru_cache
import math
from pathlib import Path

from .protocol import json_value
from .tasks import Task

SPEED_BINS = [0.5, 2, 4, 6, 8, 10]


@lru_cache(maxsize=32)
def wind_rose(path: Path) -> dict:
    # AERMET SFC fields 16/17 (one-based) are reference wind speed/direction.
    # Bin edges are lower bounds; the final bin includes speeds >= 10 m/s.
    counts = [[0] * len(SPEED_BINS) for _ in range(16)]
    calm = 0
    with path.open() as source:
        for line in source:
            fields = line.split()
            if len(fields) < 17 or not fields[0].isdigit():
                continue
            speed, direction = float(fields[15]), float(fields[16])
            if not math.isfinite(speed) or speed < 0 or speed >= 90:
                continue
            if speed < SPEED_BINS[0]:
                calm += 1
                continue
            if not math.isfinite(direction) or not 0 <= direction <= 360:
                continue
            sector = int((direction % 360 + 11.25) // 22.5) % 16
            counts[sector][bisect_right(SPEED_BINS, speed) - 1] += 1
    return {'sectors': 16, 'speed_bins_m_s': SPEED_BINS.copy(), 'counts': counts, 'calm_hours': calm}


def geometry(task: Task) -> dict:
    scenario = task.scenario
    return json_value({
        'task_id': task.task_id, 'station': scenario.station,
        'buildings': [{'id': b['id'], 'corners': b.get('corners', b.get('corners_m')), 'height_m': b['height_m']}
                      for b in scenario.buildings['buildings']],
        'receptors': scenario.receptors,
        'stacks': [{'id': s.id, 'x_m': s.location['x_m'], 'y_m': s.location['y_m'],
                    'height_min_m': s.height_min_m, 'gep_m': s.gep_height_m,
                    'diameter_m': s.fixed_stack['diameter_m']} for s in scenario.sources],
        'wind_rose': wind_rose(scenario.met_path / f'{scenario.station}.sfc'),
    })
