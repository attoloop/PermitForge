import pytest

from permit_core.native import Reject, bpip_results, run_native


def test_native_timeout(tmp_path):
    binary = tmp_path / "wait-model"
    binary.write_text("#!/bin/sh\nexec sleep 10\n")
    binary.chmod(0o755)
    with pytest.raises(Reject) as error:
        run_native(binary, tmp_path, 1, {"resources": {"native_launches": 0}})
    assert error.value.code == "E_RESOURCE"


def test_bpip_65_metre_floor(tmp_path):
    path = tmp_path / "fort.12"
    path.write_text("".join(f"SO {kind} S1 " + " ".join(["0"] * 36) + "\n"
                            for kind in ["BUILDHGT", "BUILDWID", "BUILDLEN", "XBADJ", "YBADJ"])
                    + "S1 0 0 0 12\n")
    down, gep = bpip_results(path, ["S1"])
    assert gep["S1"] == 65
    assert all(len(row) == 36 for row in down.values())


def test_case_bytes_skips_files_the_model_deletes_mid_scan(tmp_path, monkeypatch):
    from pathlib import Path
    from permit_core.native import case_bytes
    (tmp_path / "aermod.out").write_bytes(b"xx")
    ghost = tmp_path / "EVENT.TMP"  # listed while AERMOD ran, deleted by it before stat()
    real_iterdir = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda self: [*real_iterdir(self), ghost])
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    assert case_bytes(tmp_path) == 2
