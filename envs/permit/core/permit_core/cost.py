# v1 origin: tasks/permit-forge/instances/003bd328a392-s47/tests/grade.py (cost, costs)
"""Exact annualized costs and the published failure floor."""
from .native import D, dec, numeric, require

def cost(ledger, task):
    controls = {c['id']: c for c in task['costs']['controls']}
    total = D(0)
    for u in task['scenario']['units']:
        row = ledger[u['id']]
        price = task['costs']['stack_usd_per_m']
        if isinstance(price, dict):
            price = price[u['id']]
        total += (D(0) if row['control'] == 'none' else dec(controls[row['control']]['annualized_usd']))
        total += dec(price) * (dec(row['stack_height_m']) - dec(u['stack_bounds']['height_min_m']))
    return total


def costs(task, draft):
    maximum = {}
    for c in task['costs']['controls']:
        require(numeric(c['annualized_usd'], 'E_OUTPUT') >= 0, 'E_OUTPUT', 'negative bound cost')
        for efficiency in c['efficiency_by_pollutant'].values():
            require(0 <= numeric(efficiency, 'E_OUTPUT') <= 1, 'E_OUTPUT', 'bound efficiency outside [0,1]')
    for u in task['scenario']['units']:
        sid = u['id']
        price = task['costs']['stack_usd_per_m']
        require(numeric(price[sid] if isinstance(price, dict) else price, 'E_OUTPUT') >= 0, 'E_OUTPUT', 'negative bound stack price')
        available = [c for c in task['costs']['controls'] if sid in c['applies_to']]
        c = max(available, key=lambda c: dec(c['annualized_usd']), default=None)
        maximum[sid] = {'control': c['id'] if c else 'none', 'stack_height_m': u['stack_bounds']['height_max_m']}
    cd, cm = cost(draft, task), cost(maximum, task)
    return cd, cm, cd-cm-1

