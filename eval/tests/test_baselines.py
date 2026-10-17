
from eval.baselines import compute_policies
from eval.common import write_json
from eval.report import baseline_rows


def view():
    return {'task_id': 't', 'draft_cost': 100, 'evaluations_left': 30, 'stack_usd_per_m': 1,
            'sources': [{'id': 'S1', 'height_min_m': 10, 'gep_m': 20, 'height_menu_m': [10,20]}],
            'controls': [{'id': 'S1_'+c, 'applies_to': ['S1'], 'annualized_usd': price} for c,price in
                         [('none',0),('moderate',50),('maximum',90)]],
            'draft_design': {'S1': {'control': 'S1_maximum', 'height_m': 20}}}


def test_all_core_policies_verified_before_reporting():
    checked = []
    def fast(design):
        return {'valid':True, 'feasible':design['S1']['control'] != 'S1_none'}
    def verify(design):
        checked.append(design)
        return {**fast(design), 'cost': 100}
    results = compute_policies(view(), fast, verify)
    assert len(results) == 6 and len(checked) > 0
    assert all(r['feasible'] or r['usd_saved'] == -1 for r in results.values())
    assert all(r.get('verified') for r in results.values() if r['design'] is not None)


def test_report_baselines_exact_same_task_cohort(tmp_path):
    from eval.tests.test_contract import episode
    from permit_core.baselines import POLICIES
    rows = {name: {'feasible':True, 'usd_saved':10, 'evaluations_used':1, 'verified':True, 'design':{}}
            for name in POLICIES}
    write_json(tmp_path/'a'/'baselines.json', rows)
    result, missing = baseline_rows([episode()], tmp_path)
    assert len(result) == 6 and not missing and all(r['task_ids'] == ['a'] for r in result)
    result, missing = baseline_rows([episode(), episode(task='b')], tmp_path)
    assert not result and 'b' in missing
