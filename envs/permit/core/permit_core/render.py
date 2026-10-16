# v1 origin: tasks/permit-forge/instances/003bd328a392-s47/environment/render.py
# design_text: tasks/permit-forge/instances/003bd328a392-s47/solution/materialize.py
"""Public input reconstruction; all downwash comes from a fresh BPIP run."""
import json
import shlex
import tempfile
from decimal import Decimal as D, ROUND_HALF_EVEN
from pathlib import Path
from copy import deepcopy
from typing import Literal

from .validate import validate_ledger

DOWN = {'BUILDHGT', 'BUILDWID', 'BUILDLEN', 'XBADJ', 'YBADJ'}

No2Tier = Literal['PVMRM', 'ARM2', 'as_published']


def with_no2_tier(task: dict, tier: No2Tier) -> dict:
    """Copy the locked grammar, changing only the explicitly selected NO2 tier."""
    if tier not in {'PVMRM', 'ARM2', 'as_published'}:
        raise ValueError(f'Unknown NO2 tier: {tier!r}')
    result = deepcopy(task)
    if tier == 'as_published':
        return result
    rows = result['locked_keyword_table']['NO2']
    published = next(r['fields'] for r in rows if r['fields'][1] == 'MODELOPT')
    if tier == 'PVMRM' and 'PVMRM' in published:
        return result
    cleaned = []
    for row in rows:
        f = row['fields']
        if f[1] in {'OZONEVAL', 'NO2RATIO', 'ARMRATIO'}:
            continue
        if f[1] == 'MODELOPT':
            row['fields'] = [v for v in f if v not in {'PVMRM', 'ARM2'}] + [tier]
        if tier == 'PVMRM' and f[:2] == ['CO', 'FINISHED']:
            cleaned.append({'fields': ['CO', 'OZONEVAL', str(task['background']['ozone']['value']), 'PPB'],
                            'mutable_field_indices': []})
        cleaned.append(row)
        if tier == 'PVMRM' and f[:2] == ['SO', 'SRCPARAM']:
            source = next(u for u in task['scenario']['units'] if u['id'] == f[2])
            cleaned.append({'fields': ['SO', 'NO2RATIO', f[2], str(source['no2_nox_in_stack_ratio'])],
                            'mutable_field_indices': []})
    result['locked_keyword_table']['NO2'] = cleaned
    return result


def bpip_template(root, task, ledger):
    lines = (root/'draft/bpip.inp').read_text().splitlines()
    for row in task['bpip_stack_records']:
        sid = row[0]
        lines.append(shlex.join([sid, row[1],
                                format(D(str(ledger[sid]['stack_height_m'])), '.2f'),
                                *row[2:]]))
    return '\n'.join(lines)+'\n'


def render(root: Path, design: dict, permit: Path, run_bpip=None, *, task=None) -> None:
    task = task or json.loads((root/'task.json').read_text(), parse_float=D)
    ledger = validate_ledger(design, task)
    permit.mkdir(parents=True, exist_ok=True)
    (permit/'design.yaml').write_text(design_text(ledger)+'\n')
    (permit/'bpip.inp').write_text(bpip_template(root, task, ledger))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)/'bpip'
        if run_bpip is None:
            from .verify import run_bpip as native_bpip
            native_bpip(permit/'bpip.inp', out, task['guards']['native_seconds'])
        else:
            run_bpip(permit/'bpip.inp', out)
        down = {}
        key = None
        for line in (out/'fort.12').read_text().splitlines():
            fields = line.split()
            if len(fields) >= 3 and fields[0] == 'SO' and fields[1] in DOWN:
                key = (fields[1], fields[2])
                down.setdefault(key, []).extend(fields[3:])
            elif fields and key and all(is_number(v) for v in fields):
                down[key].extend(fields)
            else:
                key = None
        assert set(down) == {(k, sid) for k in DOWN for sid in ledger}
        assert all(len(v) == 36 for v in down.values())
    sources = {u['id']: u for u in task['scenario']['units']}
    controls = {c['id']: c for c in task['costs']['controls']}
    result = []
    for p in task['scenario']['pollutants']:
        result.append('** PERMIT_CASE '+p)
        offsets = {}
        for row in task['locked_keyword_table'][p]:
            f = list(row['fields'])
            if f[1] == 'SRCPARAM':
                sid = f[2]; choice = ledger[sid]['control']
                eff = D(0) if choice == 'none' else D(str(controls[choice]['efficiency_by_pollutant'][p]))
                rate = D(str(sources[sid]['uncontrolled_rates_g_s'][p]))*(1-eff)
                f[3] = format(rate.quantize(D('1e-9'), rounding=ROUND_HALF_EVEN), 'f')
                f[4] = format(ledger[sid]['stack_height_m'], '.2f')
            elif f[1] in DOWN:
                key = (f[1], f[2]); offset = offsets.get(key, 0)
                count = len(f)-3
                f[3:] = down[key][offset:offset+count]
                offsets[key] = offset+count
            result.append(' '.join(f))
    (permit/'aermod.inp').write_text('\n'.join(result)+'\n')


def is_number(value):
    try:
        return D(value).is_finite()
    except Exception:
        return False



def design_text(ledger):
    # Numeric JSON heights preserve exact decimals without a binary float roundtrip.
    return '{'+','.join(json.dumps(sid)+':{"control":'+json.dumps(row['control'])+
                       ',"stack_height_m":'+format(D(str(row['stack_height_m'])), 'f')+'}'
                       for sid, row in sorted(ledger.items()))+'}'

