"""Synthetic extraction requirements: uv run pytest -q."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from convert.extract_v1 import extract
from convert.verify_extract import verify


def put(root: Path, name: str, data: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)
    return path


@pytest.fixture
def family(tmp_path):
    src, runs, out = (tmp_path / name for name in ("src", "runs", "out"))
    rows = []
    for ident, station, split in (("a", "KAAA", "train"), ("b", "KAAA", "train"),
                                  ("c", "KBBB", "eval")):
        rows.append({"id": ident, "path": f"instances/{ident}",
                     "equivalence_key": f"meteorology:{station}",
                     "coverage_tags": [f"split:{split}"]})
        base = f"instances/{ident}/"
        for name in ("instruction.md", "task.toml", "instance_manifest.json"):
            put(src, base + name, ident)
        for name in ("environment/common.py", "tests/grade.py", "solution/solve.sh"):
            put(src, base + name, "shared")
        put(src, base + "environment/vary.py", "variant" if ident == "c" else "same")
        for task in ("environment/task", "tests/task"):
            for suffix in ("sfc", "pfl"):
                put(src, base + f"{task}/met/{station}.{suffix}", station + suffix)
            put(src, base + f"{task}/docs/guide.pdf", "guide")
            put(src, base + f"{task}/draft/design.yaml", ident + task)
            put(src, base + f"{task}/task.json", ident)
    put(src, "manifest.json", json.dumps({"instances": rows}))
    put(src, "bind.py", "family")
    put(src, "template/environment/extra.py", "template")
    put(src, "instances/a/extra.dat", "unexpected but preserved")
    put(runs, "bind_verify/1/replay-baselines/a/oracle/result.json", "{}")
    evidence = "mechanical_matrix/7355/harbor/"
    put(runs, evidence + "result.log", "evidence")
    put(runs, evidence + "excluded.bin", "binary")
    put(runs, evidence + "large.txt", "x" * (1024 * 1024 + 1))
    put(runs, evidence + "boundary.txt", "x" * (1024 * 1024))
    put(runs, evidence + "unreadable.txt", "secret").chmod(0)
    os.mkfifo(runs / evidence / "fifo.txt")
    (runs / evidence / "root").symlink_to("/", target_is_directory=True)
    (runs / evidence / "link.json").symlink_to("/etc/passwd")
    probe = runs / "adversarial_codex/7/probe-runs/fifo-probe"
    probe.mkdir(parents=True)
    os.mkfifo(probe / "aermod.inp")
    (probe / "root").symlink_to("/", target_is_directory=True)
    (probe.parent / "escape").symlink_to("/", target_is_directory=True)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "src"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "remote", "add", "origin", "fixture-origin"],
                   check=True)
    return src, runs, out


def test_round_trip_dedup_and_inventory(family):
    src, runs, out = family
    manifest = extract(src, runs, out)
    assert json.loads((out / "EXTRACT_MANIFEST.json").read_text()) == manifest
    assert manifest["source_git"]["origin"] == "fixture-origin"
    assert len(manifest["source_git"]["head"]) == 40
    assert manifest["source_git"]["branch"]
    assert manifest["instances"]["a"]["met_station"] == "KAAA"
    assert manifest["summary"]["instances"] == 3
    assert manifest["summary"]["stations"] == 2
    assert manifest["summary"]["splits"] == {"train": 2, "eval": 1}
    assert manifest["summary"]["variants"] == 1
    assert (out / "shared/environment/common.py").read_text() == "shared"
    assert not (out / "shared/environment/vary.py").exists()
    variants = manifest["variants"]["environment/vary.py"]
    assert sorted(sorted(row["instances"]) for row in variants) == [["a", "b"], ["c"]]
    for row in variants:
        assert (out / row["destination"]).name == row["sha256"][:12]
    assert (out / "met/KAAA/KAAA.sfc").read_text() == "KAAAsfc"
    assert (out / "shared/docs/guide.pdf").read_text() == "guide"
    assert (out / "instances/a/task/draft/design.yaml").read_text() == "aenvironment/task"
    assert (out / "instances/a/verifier-task/draft/design.yaml").read_text() == "atests/task"
    assert (out / "family/template/environment/extra.py").exists()
    assert any(row["source"] == "instances/a/extra.dat" for row in manifest["files"])
    expected = {p.relative_to(src).as_posix() for p in src.rglob("*")
                if p.is_file() and ".git" not in p.relative_to(src).parts}
    assert {row["source"] for row in manifest["files"] if row["root"] == "src"} == expected
    assert verify(out) == []
    assert verify(out, src) == []


def test_evidence_safety(family, monkeypatch):
    src, runs, out = family
    sock = socket.socket(socket.AF_UNIX)
    sock.bind(str(runs / "mechanical_matrix/7355/socket.txt"))
    original = os.open

    def guarded(path, flags, *args, **kwargs):
        if str(path).startswith(str(runs)):
            assert not Path(path).is_symlink()
            assert Path(path).is_file(), f"opened special file {path}"
            assert Path(path).stat().st_mode & 0o444, "opened unreadable file"
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded)
    try:
        manifest = extract(src, runs, out)
    finally:
        sock.close()
    evidence = [r for r in manifest["files"] if r["root"] == "runs"]
    assert len(evidence) == 3
    skipped = {r["source"]: r["reason"] for r in manifest["skipped"]}
    assert any("fifo.txt" in p and "regular" in reason for p, reason in skipped.items())
    assert any("socket.txt" in p and "regular" in reason for p, reason in skipped.items())
    assert any("root" in p and "symlink" in reason for p, reason in skipped.items())
    assert any("unreadable.txt" in p and "unreadable" in reason for p, reason in skipped.items())
    assert json.loads((out / "evidence/adversarial_probes.json").read_text()) == [
        {"node": "adversarial_codex", "attempt": "7", "probe": "fifo-probe"}]


@pytest.mark.parametrize("kind", ["met", "docs"])
def test_conflicting_station_or_docs_rejected(family, kind):
    src, runs, out = family
    name = "met/KAAA.sfc" if kind == "met" else "docs/guide.pdf"
    (src / "instances/b/tests/task" / name).write_text("conflict")
    with pytest.raises(ValueError, match="conflicting"):
        extract(src, runs, out)
    assert not (out / "EXTRACT_MANIFEST.json").exists()


def test_integrity_failures_and_missing_mapping(family):
    src, runs, out = family
    extract(src, runs, out)
    target = out / "shared/environment/common.py"
    target.write_text("corrupt")
    assert verify(out)
    target.write_text("shared")
    assert verify(out) == []
    put(src, "new.txt", "new")
    assert verify(out, src)
    (src / "new.txt").unlink()
    (src / "bind.py").write_text("changed")
    assert verify(out, src)
    (src / "bind.py").write_text("family")
    manifest_path = out / "EXTRACT_MANIFEST.json"
    data = json.loads(manifest_path.read_text())
    data["files"] = [r for r in data["files"] if r["source"] != "bind.py"]
    manifest_path.write_text(json.dumps(data))
    assert verify(out, src)


def test_source_specials_and_unreadable_are_reported(family):
    src, runs, out = family
    os.mkfifo(src / "fifo")
    (src / "escape").symlink_to("/", target_is_directory=True)
    put(src, "unreadable", "hidden").chmod(0)
    manifest = extract(src, runs, out)
    assert {"fifo", "escape", "unreadable"} <= {r["source"] for r in manifest["skipped"]}
    assert verify(out) == []
    assert verify(out, src)  # Cannot claim lossless when a regular source was unreadable.


def test_refuses_nonempty_output(family):
    src, runs, out = family
    put(out, "keep", "existing")
    with pytest.raises(ValueError, match="empty"):
        extract(src, runs, out)
    assert (out / "keep").read_text() == "existing"


def test_cli(family):
    src, runs, out = family
    def run(script, *args):
        return subprocess.run([sys.executable, f"convert/{script}.py", *map(str, args)],
                              capture_output=True, text=True, timeout=20)
    assert run("extract_v1", "--src", src, "--runs", runs, "--out", out).returncode == 0
    assert run("verify_extract", "--out", out, "--src", src).returncode == 0
    (out / "family/bind.py").unlink()
    assert run("verify_extract", "--out", out).returncode != 0


def test_missing_station_file_rejected(family):
    src, runs, out = family
    (src / "instances/b/tests/task/met/KAAA.pfl").unlink()
    with pytest.raises(ValueError, match="missing met"):
        extract(src, runs, out)


@pytest.mark.parametrize("kind", ["symlink_file", "symlink_directory", "traversal"])
def test_verify_rejects_unsafe_destination(family, kind):
    src, runs, out = family
    extract(src, runs, out)
    if kind == "symlink_file":
        (out / "family/bind.py").unlink()
        (out / "family/bind.py").symlink_to(src / "bind.py")
    elif kind == "symlink_directory":
        (out / "shared/environment").rename(out / "saved-environment")
        (out / "shared/environment").symlink_to(out / "saved-environment", target_is_directory=True)
    else:
        manifest_path = out / "EXTRACT_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][0]["destination"] = "../src/bind.py"
        manifest_path.write_text(json.dumps(manifest))
    assert verify(out)
