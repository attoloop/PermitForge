"""Preserve v1: uv run python convert/extract_v1.py --src SRC --runs RUNS --out OUT."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import stat
import subprocess

if __package__:
    from .extract_io import Record, digest, entries, open_regular, skip, walk
else:
    from extract_io import Record, digest, entries, open_regular, skip, walk


def git_metadata(src: Path) -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(src), *args], text=True).strip()
    return {"head": git("rev-parse", "HEAD"), "origin": git("remote", "get-url", "origin"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD")}


def directories(root: Path, relative: Path, skipped: list[Record]) -> list[Path]:
    result = []
    for entry in entries(root, relative, skipped, "runs"):
        mode = entry.stat(follow_symlinks=False).st_mode
        rel = relative / entry.name
        if stat.S_ISDIR(mode):
            result.append(rel)
        elif not stat.S_ISREG(mode):
            skip(skipped, "runs", rel.as_posix(),
                 "symlink" if stat.S_ISLNK(mode) else "not a regular file")
    return result


def extract(src: Path, runs: Path, out: Path) -> Record:
    src, runs, out = Path(src).absolute(), Path(runs).absolute(), Path(out).absolute()
    for root in (src, runs):
        if not stat.S_ISDIR(root.lstat().st_mode):
            raise ValueError(f"input must be a real directory: {root}")
    for root in (src, runs):
        if out.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(out.resolve()):
            raise ValueError("output must be separate from both inputs")
    if out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise ValueError("output must be absent or an empty directory")
    provenance = git_metadata(src)
    out.mkdir(parents=True, exist_ok=True)
    skipped: list[Record] = []
    files: list[Record] = []
    cache: dict = {}
    for relative, _ in walk(src, skipped, "src"):
        try:
            size, sha = digest(src / relative, cache)
        except (OSError, ValueError) as exc:
            skip(skipped, "src", relative, f"unreadable: {exc}")
            continue
        files.append({"root": "src", "source": relative, "size": size, "sha256": sha})

    with open_regular(src / "manifest.json") as stream:
        family = json.load(stream)
    instances = {row["id"]: {"met_station": None} for row in family["instances"]}
    shared: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in files:
        parts = Path(row["source"]).parts
        if len(parts) >= 4 and parts[0] == "instances" and parts[2] in (
                "environment", "tests", "solution") and parts[2:4] not in (
                    ("environment", "task"), ("tests", "task")):
            shared["/".join(parts[2:])][row["sha256"]].append(parts[1])

    variants: dict[str, list[Record]] = {}
    met: dict[str, dict[str, str]] = defaultdict(dict)
    met_locations: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    docs: dict[str, str] = {}
    destinations: dict[str, str] = {}

    def copy(row: Record, destination: str) -> None:
        previous = destinations.get(destination)
        if previous is not None and previous != row["sha256"]:
            raise ValueError(f"conflicting content at {destination}")
        row["destination"] = destination
        if previous is None:
            target = out / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            with open_regular((src if row["root"] == "src" else runs) / row["source"]) as source:
                with target.open("xb") as output:
                    shutil.copyfileobj(source, output)
            # Preserve executable scripts, but never reproduce mode-000 or special bits.
            mode = (src if row["root"] == "src" else runs) / row["source"]
            target.chmod(0o755 if mode.lstat().st_mode & 0o111 else 0o644)
            if digest(target) != (row["size"], row["sha256"]):
                raise ValueError(f"copy integrity failure: {row['source']}")
            destinations[destination] = row["sha256"]

    for relative, by_hash in shared.items():
        if len(by_hash) > 1:
            variants[relative] = [{"sha256": sha, "instances": sorted(set(ids)),
                                   "destination": f"variants/{relative}/{sha[:12]}"}
                                  for sha, ids in sorted(by_hash.items())]

    for row in files:
        parts = Path(row["source"]).parts
        destination = f"family/{row['source']}"
        if parts[0] == "instances" and len(parts) >= 3:
            ident = parts[1]
            instances.setdefault(ident, {"met_station": None})
            rel = "/".join(parts[2:])
            if rel in shared:
                destination = (f"variants/{rel}/{row['sha256'][:12]}" if rel in variants
                               else f"shared/{rel}")
            elif parts[2:4] in (("environment", "task"), ("tests", "task")):
                task_rel = parts[4:]
                if len(task_rel) >= 2 and task_rel[0] == "met":
                    station = Path(task_rel[-1]).stem
                    old_station = instances[ident]["met_station"]
                    if old_station is not None and old_station != station:
                        raise ValueError(f"conflicting met stations for {ident}")
                    instances[ident]["met_station"] = station
                    filename = "/".join(task_rel[1:])
                    previous = met[station].setdefault(filename, row["sha256"])
                    if previous != row["sha256"]:
                        raise ValueError(f"conflicting met file for {station}/{filename}")
                    met_locations[ident][parts[2]][filename] = row["sha256"]
                    destination = f"met/{station}/{filename}"
                elif len(task_rel) >= 2 and task_rel[0] == "docs":
                    filename = "/".join(task_rel[1:])
                    previous = docs.setdefault(filename, row["sha256"])
                    if previous != row["sha256"]:
                        raise ValueError(f"conflicting docs file: {filename}")
                    destination = f"shared/docs/{filename}"
                else:
                    folder = "task" if parts[2] == "environment" else "verifier-task"
                    destination = f"instances/{ident}/{folder}/{'/'.join(task_rel)}"
            else:
                destination = f"instances/{ident}/{rel}"
        copy(row, destination)

    for ident, data in instances.items():
        station = data["met_station"]
        if station is None:
            raise ValueError(f"missing meteorology for {ident}")
        for location in ("environment", "tests"):
            if met_locations[ident][location] != met[station]:
                raise ValueError(f"conflicting or missing met files: {ident}/{location}")

    evidence_roots = [Path("mechanical_matrix/7355")]
    for attempt in directories(runs, Path("bind_verify"), skipped):
        evidence_roots.append(attempt / "replay-baselines")
    for base in evidence_roots:
        for relative, info in walk(runs, skipped, "runs", base):
            if Path(relative).suffix not in {".json", ".txt", ".log", ".toml", ".md"}:
                skip(skipped, "runs", relative, "excluded suffix")
                continue
            if info.st_size > 1024 * 1024:
                skip(skipped, "runs", relative, "larger than 1 MiB")
                continue
            try:
                size, sha = digest(runs / relative, cache)
                row = {"root": "runs", "source": relative, "size": size, "sha256": sha}
                copy(row, f"evidence/{relative}")
                files.append(row)
            except (OSError, ValueError) as exc:
                skip(skipped, "runs", relative, f"unreadable: {exc}")

    probes = []
    for node in directories(runs, Path(), skipped):
        if not node.name.startswith("adversarial_"):
            continue
        for attempt in directories(runs, node, skipped):
            for probe in directories(runs, attempt / "probe-runs", skipped):
                probes.append({"node": node.name, "attempt": attempt.name, "probe": probe.name})
    probe_path = "evidence/adversarial_probes.json"
    (out / "evidence").mkdir(exist_ok=True)
    (out / probe_path).write_text(json.dumps(probes, indent=2) + "\n")
    size, sha = digest(out / probe_path)
    splits = Counter(tag.removeprefix("split:") for row in family["instances"]
                     for tag in row.get("coverage_tags", []) if tag.startswith("split:"))
    manifest = {"schema_version": 1, "source_git": provenance, "source_root": str(src),
                "runs_root": str(runs), "files": files, "skipped": skipped,
                "instances": instances, "variants": variants,
                "generated": [{"destination": probe_path, "size": size, "sha256": sha}],
                "summary": {"instances": len(instances), "stations": len(met),
                            "splits": dict(sorted(splits.items())),
                            "shared_files": sum(len(v) == 1 for v in shared.values()) + len(docs),
                            "variants": len(variants),
                            "variant_files": sum(len(v) for v in variants.values()),
                            "source_files": sum(r["root"] == "src" for r in files),
                            "evidence_files": sum(r["root"] == "runs" for r in files),
                            "adversarial_probes": len(probes), "skipped": len(skipped)}}
    (out / "EXTRACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("src", "runs", "out"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = extract(args.src, args.runs, args.out)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"extraction failed: {exc}\n")
    print(json.dumps(manifest["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
