# v1 origin: tasks/permit-forge/instances/003bd328a392-s47/environment/public_native.py
# Additional pure helpers: tasks/permit-forge/instances/003bd328a392-s47/environment/service.py
"""Published native card grammar, subprocess guards, and BPIP output parsing."""
import datetime as dt
import os
from pathlib import Path
import re
import resource
import shlex
import signal
import subprocess
import time
from decimal import Decimal, InvalidOperation, localcontext, ROUND_HALF_EVEN

D = Decimal

DOWN = {'BUILDHGT', 'BUILDWID', 'BUILDLEN', 'XBADJ', 'YBADJ'}

NUM = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?\Z', re.ASCII)

class Reject(Exception):
    def __init__(self, code, detail):
        self.code, self.detail = code, detail
        super().__init__(detail)

def require(condition, code, detail):
    if not condition:
        raise Reject(code, detail)

def dec(x):
    return D(str(x).replace('D', 'E').replace('d', 'e'))

def numeric(x, code):
    require(isinstance(x, (int, float, D)) and not isinstance(x, bool), code, f'numeric value required; got {x!r}')
    v = dec(x)
    require(v.is_finite(), code, f'finite value required; got {x!r}')
    return v

def equivalent(a, b, places=None):
    if NUM.fullmatch(str(b)):
        if not NUM.fullmatch(str(a)):
            return False
        x, y = dec(a), dec(b)
        if places is None:
            return x == y
        try:
            with localcontext() as ctx:
                ctx.prec = 50
                q = D(1).scaleb(-places)
                return x.quantize(q, rounding=ROUND_HALF_EVEN) == y.quantize(q, rounding=ROUND_HALF_EVEN)
        except InvalidOperation:
            return False
    return a == b

def native_lines(text, code):
    lines = text.replace('\r\n', '\n').split('\n')
    for line in lines:
        # Native comment bodies are opaque, not numeric or structural input.
        if line.startswith('**') and code == 'E_LOCKED':
            continue
        if code == 'E_BPIP_MISMATCH':
            quote = None
            for i, char in enumerate(line):
                if char in "'\"":
                    if quote is None:
                        quote = char
                    elif quote == char:
                        quote = None
                if char == '!' and quote is None:
                    line = line[:i]
                    break
        invalid = re.search(r'[^\x20-\x7e]' if code == 'E_LOCKED'
                            else r'[^\t\x20-\x7e]', line)
        require(invalid is None, code, 'invalid native record character')
    return lines

def bpip_records(text, with_comments=False):
    records = []
    token = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|[^ \t,/'\"\\]+")
    for line in native_lines(text, 'E_BPIP_MISMATCH'):
        fields = []
        pos = 0
        while pos < len(line):
            if line[pos] in ' \t':
                pos += 1
                continue
            if line[pos] == '!':
                if with_comments:
                    fields.append(None)  # Preserve where native READ would encounter '!'.
                break
            match = token.match(line, pos)
            require(match is not None, 'E_BPIP_MISMATCH', 'invalid list-directed token')
            value = match.group()
            pos = match.end()
            require(pos == len(line) or line[pos] in ' \t,!',
                    'E_BPIP_MISMATCH', 'invalid list-directed delimiter')
            if value[0] in "'\"":
                quote = value[0]
                decoded = value[1:-1].replace(quote*2, quote)
                # Quoted numeric fields are strings, not native numbers.
                value = value if NUM.fullmatch(decoded) else decoded
            fields.append(value)
            while pos < len(line) and line[pos] in ' \t':
                pos += 1
            if pos < len(line) and line[pos] == ',':
                pos += 1
                require(not line[pos:].lstrip(' \t').startswith(','),
                        'E_BPIP_MISMATCH', 'null list-directed field')
        if fields:
            records.append(fields)
    return records

def bpip_input(text, baseline, ledger):
    records, trusted = bpip_records(text, with_comments=True), bpip_records(baseline)
    # A Fortran READ may continue onto another record, but discards any
    # remaining fields on the record where it finishes. Never flatten those.
    cursor = 0
    for expected in trusted:
        fields = []
        while len(fields) < len(expected) and cursor < len(records):
            fields.extend(records[cursor]); cursor += 1
            if fields and fields[-1] is None:
                fields.pop()
                require(len(fields) == len(expected), 'E_BPIP_MISMATCH',
                        'BPIP comment before final item of native read')
        require(len(fields) == len(expected), 'E_BPIP_MISMATCH', 'BPIP read record field count differs')
    require(all(r == [None] for r in records[cursor:]), 'E_BPIP_MISMATCH', 'extra BPIP records')
    a, b = [v for r in records for v in r if v is not None], [v for r in trusted for v in r]
    require(len(a) == len(b), 'E_BPIP_MISMATCH', f'BPIP field count {len(a)} != {len(b)}')
    integers = {6}
    i = 7
    for _ in range(int(b[6])):
        integers.add(i+1)
        tiers = int(b[i+1]); i += 3
        for _ in range(tiers):
            integers.add(i)
            corners = int(b[i]); i += 2 + 2*corners
    integers.add(i)
    n = int(b[i]); i += 1
    for j in integers:
        require(re.fullmatch(r'[+-]?[0-9]+', a[j]) is not None,
                'E_BPIP_MISMATCH', f'BPIP integer field {j}: {a[j]!r}')
    mutable = {}
    for _ in range(n):
        mutable[i+2] = ledger[b[i]]['stack_height_m']
        i += 5
    require(i == len(b), 'E_OUTPUT', 'bound BPIP grammar incomplete')
    for j, (got, want) in enumerate(zip(a, b)):
        if j in mutable:
            precision = max(0, -dec(b[j]).as_tuple().exponent)
            height = dec(mutable[j])
            require(height == height.quantize(D(1).scaleb(-precision), rounding=ROUND_HALF_EVEN),
                    'E_BPIP_MISMATCH', f'BPIP design height {height} is not representable at {precision} decimal places')
            require(equivalent(got, str(height)), 'E_BPIP_MISMATCH', f'BPIP height field {j}: {got} != {height}')
            b[j] = str(mutable[j])
        else:
            require(equivalent(got, want), 'E_BPIP_MISMATCH', f'BPIP field {j}: {got!r} != {want!r}')
    # Use the trusted template's record structure, replacing only validated heights.
    lines = baseline.splitlines()
    for idx, line in enumerate(lines):
        t = shlex.split(line)
        if len(t) == 5 and t[0] in ledger:
            t[0] = repr(t[0]); t[2] = str(ledger[shlex.split(line)[0]]['stack_height_m'])
            lines[idx] = ' '.join(t)
    return '\n'.join(lines)+'\n'

def cards(text):
    rows = []
    for line in native_lines(text, 'E_LOCKED'):
        if not line.strip(' ') or line.startswith('**'):
            continue
        require(len(line) <= 512 and line[:2] in {'CO', 'SO', 'RE', 'ME', 'OU', 'EV'},
                'E_LOCKED', 'invalid native pathway columns or record exceeds 512 columns')
        t = re.split(r' +', line.rstrip(' '))
        require(len(t) >= 2, 'E_LOCKED', f'incomplete native record {line!r}')
        require(all(len(v) <= 200 for v in t), 'E_LOCKED', 'native field exceeds 200 columns')
        for value in t[2:]:
            if NUM.fullmatch(value) and re.search('[eEdD]', value):
                # AERMOD STONUM/STODBL accept E after a decimal point,
                # with an exponent magnitude no greater than 30. PMARK is
                # not reset at E, so a leading + forbids an exponent +.
                # NMARK is reset: a negative mantissa permits either sign.
                parts = re.split('[eEdD]', value)
                require('d' not in value.lower() and '.' in parts[0] and
                        abs(dec(parts[1])) <= 30 and value.count('+') <= 1,
                        'E_LOCKED', f'invalid AERMOD exponent spelling {value!r}')
        if len(t) >= 3 and t[1] in DOWN and rows and rows[-1][:3] == t[:3]:
            rows[-1].extend(t[3:])
        else:
            rows.append(t)
    return rows

def cases(text, task, ledger, down, unit_response=None):
    blocks = []
    for line in native_lines(text, 'E_LOCKED'):
        if line.startswith('** PERMIT_CASE '):
            p = line.removeprefix('** PERMIT_CASE ')
            blocks.append((p, []))
        else:
            require(not re.match(r'^\*\* PERMIT_(?:CASE|HOURLY)(?:[ \t]|$)', line),
                    'E_LOCKED', f'invalid reserved marker {line[:100]}')
            require(bool(blocks) or not line.strip() or line.strip().startswith('**'), 'E_LOCKED', 'native content before case delimiter')
            if blocks:
                blocks[-1][1].append(line)
    require([p for p, _ in blocks] == task['scenario']['pollutants'], 'E_LOCKED', f'case order {[p for p, _ in blocks]} != {task["scenario"]["pollutants"]}')
    units = {u['id']: u for u in task['scenario']['units']}
    controls = {c['id']: c for c in task['costs']['controls']}
    result = []
    parsed = []
    for p, lines in blocks:
        got = cards('\n'.join(lines))
        expected = cards('\n'.join(' '.join(r['fields']) for r in task['locked_keyword_table'][p]))
        # Complete stage 2 across ALL cases before any stage-3 check.
        mutable = DOWN | {'SRCPARAM'}
        fixed_got = [r for r in got if r[1] not in mutable]
        fixed_expected = [r for r in expected if r[1] not in mutable]
        require(len(fixed_got) == len(fixed_expected) and
                all(len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
                    for a, b in zip(fixed_got, fixed_expected)),
                'E_LOCKED', f'{p}: locked records differ')
        source_expected = {r[2]: r for r in expected if r[1] == 'SRCPARAM'}
        for r in got:
            if r[1] in mutable:
                require(r[0] == 'SO', 'E_LOCKED', f'{p}: invalid source pathway {r[:3]}')
            if r[1] == 'SRCPARAM' and len(r) >= 3 and r[2] in source_expected:
                b = source_expected[r[2]]
                require(all(equivalent(x, y) for x, y in zip(r[5:], b[5:])),
                        'E_LOCKED', f'{p}: fixed source parameters differ for {r[2]}')
        # Missing/extra mutable records are stage 3, but a complete sequence
        # must retain the published structural record order.
        if (len(got) == len(expected) and
                sorted(tuple(r[:3]) for r in got if r[1] in mutable) ==
                sorted(tuple(r[:3]) for r in expected if r[1] in mutable)):
            require(all(a[:2] == b[:2] for a, b in zip(got, expected)),
                    'E_LOCKED', f'{p}: record order differs')
        parsed.append((p, got, expected))
    for p, got, expected in parsed:
        require(sorted((r[1], r[2]) for r in got if r[1] in mutable and len(r) >= 3) ==
                sorted((r[1], r[2]) for r in expected if r[1] in mutable),
                'E_DESIGN_MISMATCH', f'{p}: missing or duplicate source/downwash records')
        require(len(got) == len(expected), 'E_DESIGN_MISMATCH', f'{p}: {len(got)} records != {len(expected)}')
        for a, b in zip(got, expected):
            require(len(a) == len(b), 'E_DESIGN_MISMATCH' if b[1] in mutable else 'E_LOCKED', f'{p}: record {a[:3]} != expected {b[:3]} or field count differs')
            for j, (x, y) in enumerate(zip(a, b)):
                code, precision = ('E_DESIGN_MISMATCH' if b[1] in mutable and j == 2 else 'E_LOCKED'), None
                if b[1] == 'SRCPARAM' and j in {3, 4}:
                    sid = b[2]; row = ledger[sid]
                    if j == 3:
                        eff = D(0) if row['control'] == 'none' else dec(controls[row['control']]['efficiency_by_pollutant'][p])
                        y = str(dec(units[sid]['uncontrolled_rates_g_s'][p])*(1-eff))
                        if unit_response is not None:
                            y = '1' if sid == unit_response else '0'
                        precision = 9
                    else:
                        precision = max(0, -dec(y).as_tuple().exponent)
                        y = str(row['stack_height_m'])
                        require(dec(y) == dec(y).quantize(D(1).scaleb(-precision), rounding=ROUND_HALF_EVEN),
                                'E_DESIGN_MISMATCH', f'{sid}: design height {y} is not representable at {precision} decimal places')
                    code = 'E_DESIGN_MISMATCH'
                elif b[1] in DOWN and j >= 3:
                    y = str(down[(b[1], b[2])][j-3]); precision = 2
                    code = 'E_DESIGN_MISMATCH'
                comparison_precision = None if b[1] == 'SRCPARAM' and j == 4 else precision
                require(equivalent(x, y, comparison_precision), code, f'{p} {b[:3]} field {j}: {x} != {y}')
                if precision is not None:
                    b[j] = format(dec(y).quantize(D(1).scaleb(-precision), rounding=ROUND_HALF_EVEN), 'f')
        native = []
        for r in expected:
            if r[1] in DOWN:
                native.extend(' '.join(r[:3]+r[i:i+6]) for i in range(3,len(r),6))
            else:
                native.append(' '.join(r))
        native = [(lambda f: f'{f[0]} {f[1]:<8} '+ ' '.join(f[2:]))(line.split()) for line in native]
        result.append((p, '\n'.join(native)+'\n'))
    return result

def child_limits(seconds):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (4096*2**20, 4096*2**20))
    signal.alarm(seconds)

def run_native(binary, cwd, guard, report, disk_limit=20480*2**20):
    with (cwd / 'console.txt').open('wb') as log:
        with subprocess.Popen([str(binary)], cwd=cwd, stdout=log, stderr=log,
                              env={'LANG':'C', 'LC_ALL':'C', 'TZ':'UTC', 'OMP_NUM_THREADS':'1'},
                              start_new_session=True, preexec_fn=lambda: child_limits(guard)) as child:
            start = time.monotonic()
            try:
                while child.poll() is None:
                    require(time.monotonic()-start < guard, 'E_RESOURCE', f'{binary.name} wall guard reached {guard} seconds')
                    rss = 0
                    for pid in [os.getpid(), child.pid]:
                        try:
                            rss += int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                        except FileNotFoundError:
                            pass
                    report['resources']['peak_live_rss_bytes'] = max(rss, report['resources'].get('peak_live_rss_bytes', 0))
                    require(rss <= 5120*2**20, 'E_RESOURCE', f'live RSS {rss} > {5120*2**20}')
                    size = sum(f.stat().st_size for f in cwd.iterdir() if f.is_file())
                    require(size <= disk_limit, 'E_RESOURCE', f'case disk bytes {size} > {disk_limit}')
                    time.sleep(.1)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            require(child.returncode not in {-signal.SIGALRM, -signal.SIGKILL, -signal.SIGXCPU}, 'E_RESOURCE', f'{binary.name} terminated by resource signal {child.returncode}')
            require(child.returncode == 0, 'E_AERMOD_FATAL', f'{binary.name} exit {child.returncode}; expected 0')
    report['resources']['native_launches'] += 1

def bpip_results(path, sources):
    require(path.is_file(), 'E_OUTPUT', 'missing BPIP fort.12')
    text = path.read_text(errors='replace')
    down = {}
    for row in cards('\n'.join(l.lstrip() for l in text.splitlines() if l.strip().startswith('SO '))):
        if row[1] in DOWN:
            key = (row[1], row[2])
            require(key not in down and len(row[3:]) == 36, 'E_OUTPUT', f'BPIP {key}: expected unique 36 directions')
            down[key] = [dec(v) for v in row[3:]]
    require(set(down) == {(k, s) for k in DOWN for s in sources}, 'E_OUTPUT', 'BPIP directional array membership incomplete')
    gep = {}
    for line in text.splitlines():
        t = line.split()
        if len(t) == 5 and t[0] in sources:
            require(t[0] not in gep, 'E_OUTPUT', f'duplicate GEP row {t[0]}')
            gep[t[0]] = max(D(65), dec(t[-1]))
    require(set(gep) == set(sources), 'E_OUTPUT', 'GEP table incomplete')
    return down, gep

def calendar(task, p):
    years = task['scenario']['locked_keywords']['meteorology_years']
    start = dt.datetime(years[0], 1, 1, tzinfo=dt.timezone.utc)
    hours = task['screening'][p]['averaging_hours']
    return [start+dt.timedelta(hours=(i+1)*hours) for i in range(task['screening'][p]['periods'])]


def parse_blocks(text, task):
    blocks, profiles = [], set()
    for line in native_lines(text, 'E_LOCKED'):
        if line.startswith('** PERMIT_CASE '):
            blocks.append((line[15:], []))
        elif line == '** PERMIT_HOURLY':
            require(blocks and not blocks[-1][1] and blocks[-1][0] not in profiles,
                    'E_LOCKED', 'PERMIT_HOURLY must immediately follow one case delimiter')
            profiles.add(blocks[-1][0])
            blocks[-1][1].append('** screening requested')
        else:
            require(not re.match(r'^\*\* PERMIT_(?:CASE|HOURLY)(?:[ \t]|$)', line) and
                    (blocks or not line.strip() or line.lstrip().startswith('**')), 'E_LOCKED', 'invalid case marker')
            if blocks:
                blocks[-1][1].append(line)
    names = [p for p, _ in blocks]
    require(names and names == [p for p in task['scenario']['pollutants'] if p in names],
            'E_LOCKED', 'expected ordered unique scenario pollutants')
    return blocks, profiles

