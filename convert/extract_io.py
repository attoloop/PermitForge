"""Safe regular-file inventory helpers used by the extraction command-line tools."""

from collections.abc import Iterator
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, BinaryIO

Record = dict[str, Any]


def skip(skipped: list[Record], root: str, source: str, reason: str) -> None:
    skipped.append({"root": root, "source": source, "reason": reason})


def entries(root: Path, relative: Path, skipped: list[Record], label: str) -> list[os.DirEntry[str]]:
    """List a directory only after checking every component without following links."""
    current = root
    for part in ("", *relative.parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
            if not stat.S_ISDIR(mode):
                reason = "symlink" if stat.S_ISLNK(mode) else "not a directory"
                skip(skipped, label, current.relative_to(root).as_posix(), reason)
                return []
            if mode & 0o555 == 0:
                raise PermissionError("directory has no read/search permissions")
        except FileNotFoundError:
            return []
        except OSError as exc:
            skip(skipped, label, current.relative_to(root).as_posix(), f"unreadable: {exc}")
            return []
    try:
        with os.scandir(current) as scan:
            return sorted(scan, key=lambda entry: entry.name)
    except OSError as exc:
        skip(skipped, label, relative.as_posix(), f"unreadable: {exc}")
        return []


def walk(root: Path, skipped: list[Record], label: str,
         relative: Path = Path()) -> Iterator[tuple[str, os.stat_result]]:
    for entry in entries(root, relative, skipped, label):
        rel = relative / entry.name
        try:
            info = entry.stat(follow_symlinks=False)
        except OSError as exc:
            skip(skipped, label, rel.as_posix(), f"unreadable: {exc}")
            continue
        if stat.S_ISDIR(info.st_mode):
            yield from walk(root, skipped, label, rel)
        elif stat.S_ISREG(info.st_mode):
            yield rel.as_posix(), info
        else:
            reason = "symlink" if stat.S_ISLNK(info.st_mode) else "not a regular file"
            skip(skipped, label, rel.as_posix(), reason)


def open_regular(path: Path) -> BinaryIO:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"not a regular file: {path}")
    if info.st_mode & 0o444 == 0:
        raise PermissionError(f"unreadable file: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    actual = os.fstat(fd)
    if not stat.S_ISREG(actual.st_mode) or (info.st_dev, info.st_ino) != (
            actual.st_dev, actual.st_ino):
        os.close(fd)
        raise ValueError(f"file changed during open: {path}")
    return os.fdopen(fd, "rb")


def digest(path: Path, cache: dict | None = None) -> tuple[int, str]:
    info = path.lstat()
    key = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o444 == 0:
        # Keep the same safety checks even when a hardlink's hash is cached.
        with open_regular(path):
            pass
    if cache is not None and key in cache:
        return info.st_size, cache[key]
    hasher = hashlib.sha256()
    size = 0
    with open_regular(path) as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
            size += len(chunk)
        after = os.fstat(stream.fileno())
        after_key = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if after_key != key:
            raise ValueError(f"file changed while hashing: {path}")
    value = hasher.hexdigest()
    if cache is not None:
        cache[key] = value
    return size, value


def checked_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise ValueError(f"invalid relative path: {relative}")
    current = root
    for part in rel.parts[:-1]:
        current = current / part
        if not stat.S_ISDIR(current.lstat().st_mode):
            raise ValueError(f"not a real directory: {current}")
    return root / rel
