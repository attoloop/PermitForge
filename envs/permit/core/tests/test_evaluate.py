from decimal import Decimal as D

import numpy as np
import pytest

from permit_core.menu import height_menu
from permit_core.evaluate import arm2, annual_statistics, rounded, evaluate
from permit_core.fields import (FieldPack, prune_candidates, save_pack, load_pack, read_postfile,
                                certified_prune)


def test_menu():
    assert height_menu(10, 80) == [D(x) for x in (10, 20, 30, 40, 50, 60, 70, 80)]
    assert height_menu('10.01', '10.02') == [D('10.01'), D('10.02')]
    assert height_menu(65, 65) == [D(65)]
    for low, high in [(True, 65), ('NaN', 65), (66, 65)]:
        with pytest.raises(ValueError):
            height_menu(low, high)


def test_known_ranks_and_yearly_forms():
    # Ten constant days per synthetic year: eighth=3, fourth=7, second block=10.
    values = np.repeat(np.arange(1, 11, dtype=float), 24)[:, None]
    valid = np.ones(len(values), dtype=bool)
    stats = annual_statistics(values, valid, (0, 240), 'SO2')
    assert stats['annual'][0, 0] == 5.5
    assert stats['daily'][0, 0] == 7
    assert stats['three'][0, 0] == 10
    assert annual_statistics(values, valid, (0, 240), 'NO2')['daily'][0, 0] == 3
    assert annual_statistics(values, valid, (0, 240), 'PM25')['daily'][0, 0] == 3


def test_calms_denominators_and_zero_days():
    values = np.full((240, 1), 9.0)
    valid = np.zeros(240, dtype=bool)
    valid.reshape(10, 24)[:, :2] = True
    stats = annual_statistics(values, valid, (0, 240), 'PM25')
    assert stats['annual'][0, 0] == 9  # Annual divides by valid hours only.
    assert stats['daily'][0, 0] == 1  # 18 / max(2, 18)
    assert annual_statistics(values, valid, (0, 240), 'SO2')['three'][0, 0] == 6
    with pytest.raises(ValueError, match='valid'):
        annual_statistics(values, np.zeros(240, bool), (0, 240), 'NO2')


def test_arm2_after_source_sum_before_averaging():
    x = np.array([0, 100, 300, 1000, 5000.0])
    ratio = np.clip(-1.1723e-17*x**6 + 4.2795e-14*x**5 - 5.8345e-11*x**4
                    + 3.4555e-8*x**3 - 5.6062e-6*x**2 - 2.7383e-3*x + 1.2441, .5, .9)
    np.testing.assert_allclose(arm2(x), x*ratio, rtol=1e-14)
    assert arm2(np.array([300.0]))[0] != 2*arm2(np.array([150.0]))[0]


def test_rounding():
    assert rounded(D('1.234565'), 'decimal') == D('1.23456')
    assert rounded(D('1.234575'), 'decimal') == D('1.23458')
    assert rounded(1.23456789, 'plot') == D('1.23457')


def test_pack_roundtrip_and_bound(scenario, tmp_path):
    rng = np.random.default_rng(173)
    fields = {(s.id, str(h)): rng.uniform(0, .001, (240, 184))
              for s in scenario.sources for h in (s.height_min_m, s.gep_height_m)}
    pack = FieldPack(scenario.id, fields, np.ones(240, bool), (0, 240), np.arange(184))
    candidates, bounds = prune_candidates(scenario, pack)
    assert candidates.all()
    # For any choices of heights/controls the conservative bound dominates hourly sums.
    for _ in range(10):
        for standard in scenario.standards:
            total = sum(float(s.uncontrolled_rates_g_s[standard.pollutant]) * rng.uniform()
                        * fields[s.id, str(rng.choice([s.height_min_m, s.gep_height_m]))]
                        for s in scenario.sources)
            assert np.all(total.max(axis=0) <= bounds[standard.id])
    path = tmp_path / 'pack.npz'
    save_pack(pack, path, dtype='float64')
    restored = load_pack(path)
    assert restored.task_id == pack.task_id
    for key in fields:
        np.testing.assert_array_equal(fields[key], restored.fields[key])
    assert load_pack(path) is restored


def test_evaluation_contract_and_invalid_is_not_evaluated(scenario):
    fields = {(s.id, str(s.gep_height_m)): np.full((240, 184), .001)
              for s in scenario.sources}
    pack = FieldPack(scenario.id, fields, np.ones(240, bool), (0, 240), np.arange(184))
    design = {s.id: {'control': 'none', 'height_m': float(s.gep_height_m)} for s in scenario.sources}
    result = evaluate(scenario, design, pack)
    assert result['valid'] and result['feasible']
    assert len(result['standards']) == len(scenario.standards)
    assert result['standards'][0]['binding_receptor']['index'] == 1
    assert evaluate(scenario, {}, pack)['valid'] is False
    # A feasibility-safe pruning certificate alone cannot preserve the binding value.
    incomplete = FieldPack(scenario.id, {k: v[:, :1] for k, v in fields.items()},
                           pack.valid_hours, pack.year_offsets, np.array([0]))
    with pytest.raises(ValueError, match='receptor'):
        evaluate(scenario, design, incomplete)


def test_unformatted_precision_and_markers(tmp_path):
    import struct
    path = tmp_path / 'post.bin'
    body = struct.pack('<ii8s2d', 20010101, 1, b'ALL     ', 1.123456789012345, 0)
    record = struct.pack('<i', len(body)) + body + struct.pack('<i', len(body))
    path.write_bytes(record)
    dates, values = read_postfile(path, 2)
    assert dates.tolist() == [20010101]
    assert values[0, 0] == 1.123456789012345
    path.write_bytes(record[:-1])
    with pytest.raises(ValueError):
        read_postfile(path, 2)


def test_binding_certificate_preserves_peak_even_when_every_receptor_passes(scenario):
    fields = {}
    for source in scenario.sources:
        for height in height_menu(source.height_min_m, source.gep_height_m):
            values = np.full((240, 184), 1e-10)
            values[:, 8] = .01
            fields[source.id, str(height)] = values
    pack = FieldPack(scenario.id, fields, np.ones(240, bool), (0, 240), np.arange(184))
    assert prune_candidates(scenario, pack)[0].all()
    reduced, certificate = certified_prune(scenario, pack)
    assert reduced.receptor_indices.tolist() == [8]
    assert certificate['pruned'] == 183
    design = {s.id: {'control': s.id+'_maximum', 'height_m': float(s.gep_height_m)} for s in scenario.sources}
    assert evaluate(scenario, design, pack) == evaluate(scenario, design, reduced)
