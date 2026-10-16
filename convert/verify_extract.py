"""Verify v1: uv run python convert/verify_extract.py --out OUT [--src SRC]."""

import argparse
import json
from pathlib import Path

if __package__:
    from .extract_io import Record, checked_path, digest, open_regular, walk
else:
    from extract_io import Record, checked_path, digest, open_regular, walk


def verify(out: Path, src: Path | None = None) -> list[str]:
    errors: list[str] = []
    try:
        with open_regular(out / "EXTRACT_MANIFEST.json") as stream:
            manifest = json.load(stream)
        rows = manifest["files"] + manifest["generated"]
        seen: dict[str, tuple[int, str]] = {}
        cache: dict = {}
        for row in rows:
            destination = row["destination"]
            expected = (row["size"], row["sha256"])
            if destination in seen:
                if seen[destination] != expected:
                    errors.append(f"conflicting destination records: {destination}")
                continue
            seen[destination] = expected
            try:
                if digest(checked_path(out, destination), cache) != expected:
                    errors.append(f"destination hash/size mismatch: {destination}")
            except (OSError, ValueError) as exc:
                errors.append(f"destination {destination}: {exc}")
        if src is not None:
            mapped = {r["source"]: r for r in manifest["files"] if r["root"] == "src"}
            if len(mapped) != sum(r["root"] == "src" for r in manifest["files"]):
                errors.append("duplicate source mappings")
            skipped: list[Record] = []
            found = set()
            for relative, _ in walk(src, skipped, "src"):
                found.add(relative)
                if relative not in mapped:
                    errors.append(f"unmapped source file: {relative}")
                    continue
                row = mapped[relative]
                try:
                    if digest(checked_path(src, relative), cache) != (row["size"], row["sha256"]):
                        errors.append(f"source hash/size mismatch: {relative}")
                except (OSError, ValueError) as exc:
                    errors.append(f"source {relative}: {exc}")
            for relative in sorted(mapped.keys() - found):
                errors.append(f"source missing or not regular: {relative}")
            # A symlink or special file can be skipped; an unreadable directory hides files.
            errors.extend(f"source inventory incomplete: {r['source']}: {r['reason']}"
                          for r in skipped if r["reason"].startswith("unreadable"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"invalid extraction manifest: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--src", type=Path)
    args = parser.parse_args()
    errors = verify(args.out, args.src)
    if errors:
        for error in errors:
            print(error)
        print(f"FAIL: {len(errors)} verification errors")
        return 1
    print("OK: destination integrity" + (" and lossless source coverage" if args.src else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
