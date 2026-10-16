# v1 origin: tasks/permit-forge/instances/003bd328a392-s47/solution/validation.py
"""Published strict ledger parser and regulatory statistics."""
from decimal import InvalidOperation
import yaml
from .native import D, Reject, require, dec, numeric


def validate_ledger(value: object, task: dict, *, height_menu: dict | None = None) -> dict:
    """Dictionary entry point; preserve exact decimals and reject unrenderable heights."""
    units = task['scenario']['units']
    require(isinstance(value, dict) and set(value) == {u['id'] for u in units},
            'E_MENU', 'source IDs must exactly equal the published source set')
    controls = {c['id']: c for c in task['costs']['controls']}
    ledger = {}
    for u in units:
        sid = u['id']
        row = value[sid]
        require(isinstance(row, dict) and set(row) == {'control', 'stack_height_m'},
                'E_MENU', f'{sid}: required keys control,stack_height_m only')
        c = row['control']
        require(isinstance(c, str) and (c == 'none' or c in controls and sid in controls[c]['applies_to']),
                'E_MENU', f'{sid}: incompatible control {c!r}')
        h = numeric(row['stack_height_m'], 'E_MENU')
        lo, hi = (dec(u['stack_bounds'][k]) for k in ['height_min_m', 'height_max_m'])
        require(lo <= h <= hi, 'E_MENU', f'{sid}: height {h} outside [{lo}, {hi}]')
        require(h == h.quantize(D('.01')), 'E_DESIGN_MISMATCH',
                f'{sid}: design height {h} is not representable at 2 decimal places')
        if height_menu is not None:
            require(h in [dec(v) for v in height_menu[sid]], 'E_MENU', f'{sid}: height {h} off menu')
        ledger[sid] = {'control': c, 'stack_height_m': h}
    return ledger

class StrictLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        require(not self.check_event(yaml.AliasEvent), 'E_MENU', 'YAML aliases forbidden: expected zero aliases')
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        result = {}
        for kn, vn in node.value:
            key = self.construct_object(kn, deep=deep)
            require(isinstance(key, str) and key not in result, 'E_MENU', f'duplicate/non-string YAML key {key!r}')
            result[key] = self.construct_object(vn, deep=deep)
        return result


def design(text, task):
    try:
        value = yaml.load(text, Loader=StrictLoader)
    except (yaml.YAMLError, ValueError, InvalidOperation, RecursionError) as e:
        raise Reject('E_MENU', f'invalid YAML: {str(e)[:250]}')
    units = task['scenario']['units']
    require(isinstance(value, dict) and set(value) == {u['id'] for u in units}, 'E_MENU', 'source IDs must exactly equal the published source set')
    controls = {c['id']: c for c in task['costs']['controls']}
    for u in units:
        sid = u['id']
        row = value[sid]
        require(isinstance(row, dict) and set(row) == {'control', 'stack_height_m'}, 'E_MENU', f'{sid}: required keys control,stack_height_m only')
        c = row['control']
        require(isinstance(c, str) and (c == 'none' or c in controls and sid in controls[c]['applies_to']), 'E_MENU', f'{sid}: incompatible control {c!r}')
        h = numeric(row['stack_height_m'], 'E_MENU')
        lo, hi = (dec(u['stack_bounds'][k]) for k in ['height_min_m', 'height_max_m'])
        require(lo <= h <= hi, 'E_MENU', f'{sid}: height {h} outside [{lo}, {hi}]')
        row['stack_height_m'] = h
    return value


def plot_values(directory, pollutant, task):
    receptors = [(dec(r['fields'][2]), dec(r['fields'][3]))
                 for r in task['locked_keyword_table'][pollutant]
                 if r['fields'][:2] == ['RE', 'DISCCART']]
    result = {}
    for period in task['scenario']['averaging_times'][pollutant]:
        native_period = period.removesuffix('h').upper()
        filename = 'annual.plt' if period == 'annual' else f'period_{native_period}.plt'
        p = directory/filename
        require(p.is_file(), 'E_OUTPUT', f'{pollutant}: missing {filename}')
        rows = {}
        for line in p.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('*'):
                continue
            t = line.split()
            require(len(t) >= 9 and t[7] == 'ALL', 'E_OUTPUT', f'{filename}: invalid plot record {line[:100]}')
            xy = (dec(t[0]), dec(t[1]))
            val = dec(t[2])
            require(xy not in rows and xy in receptors and val.is_finite() and val >= 0,
                    'E_OUTPUT', f'{filename}: duplicate, unknown or nonfinite receptor {xy}')
            expected_period = 'ANNUAL' if period == 'annual' else f'{native_period}-HR'
            require(t[6] == expected_period, 'E_OUTPUT', f'{filename}: period {t[6]} != {expected_period}')
            rows[xy] = val
        require(set(rows) == set(receptors), 'E_OUTPUT', f'{filename}: {len(rows)} receptors != {len(receptors)}')
        result[period] = [rows[xy] for xy in receptors]
    return result


def statistics(path, pollutant, task):
    require(path.is_file(), 'E_OUTPUT', f'{pollutant}: missing native statistics collector output')
    count = task['screening'][pollutant]['receptors']
    kinds = {'ANNUAL', 'DAILY'} | ({'THREE'} if '3h' in task['scenario']['averaging_times'][pollutant] else set())
    values = {}
    for line in path.read_text().splitlines():
        t = line.split()
        require(len(t) == 4, 'E_OUTPUT', f'malformed statistics row {line[:100]}')
        kind, year, rec, val = t
        key = (kind, int(year), int(rec))
        require(key not in values and kind in kinds and 1 <= key[1] <= 5 and 1 <= key[2] <= count, 'E_OUTPUT', f'duplicate/unexpected statistic {key}')
        v = dec(val)
        require(v.is_finite() and v >= 0, 'E_OUTPUT', f'invalid concentration {val}')
        values[key] = v
    require(len(values) == len(kinds)*5*count, 'E_OUTPUT', f'{pollutant}: {len(values)} statistics != {len(kinds)*5*count}')
    plots = plot_values(path.parent, pollutant, task)
    result = {}
    for name, constraint in task['scenario']['naaqs_thresholds'].items():
        if constraint['pollutant'] != pollutant:
            continue
        form = constraint['form']
        kind, aggregate = {
            'max_receptor_mean_of_5_annual_8th_daily_maxima': ('DAILY', 'mean'),
            'max_receptor_mean_of_5_annual_4th_daily_maxima': ('DAILY', 'mean'),
            'max_receptor_mean_of_5_annual_8th_daily_means': ('DAILY', 'mean'),
            'max_receptor_mean_of_5_annual_means': ('ANNUAL', 'mean'),
            'max_receptor_max_annual_mean': ('ANNUAL', 'max'),
            'max_receptor_max_annual_second_highest_3h_block': ('THREE', 'max'),
        }[form]
        result[name] = []
        for r in range(1, count+1):
            seq = [values[kind, y, r] for y in range(1, 6)]
            val = (plots[constraint['averaging_time']][r-1] if aggregate == 'mean'
                   else max(seq).quantize(D('.00001')))
            if pollutant == 'SO2' and kind == 'DAILY':
                val = (sum(seq)/5).quantize(D('.00001'))
            bg = dec(task['background']['design_values'][name]); limit = dec(constraint['limit'])
            result[name].append({'receptor':r, 'modeled':str(val), 'background':str(bg), 'total':str(val+bg), 'limit':str(limit), 'margin':str(limit-val-bg)})
    return result


def admit_air_quality(tables):
    for name, rows in tables.items():
        for row in rows:
            require(dec(row['margin']) >= 0, 'E_NAAQS',
                    f'{name} receptor {row["receptor"]}: total {row["total"]} > limit {row["limit"]}')

