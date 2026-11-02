"""Atomic durable transitions and verified volatile metadata for Titanic Flash."""
import json
import os
from pathlib import Path
import stat
import tempfile


def atomic_bytes(path, value, durable=True):
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as output:
            output.write(value)
            output.flush()
            if durable:
                os.fsync(output.fileno())
        temporary.replace(path)
        if durable:
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def atomic(path, value, durable=True):
    atomic_bytes(path, (json.dumps(value, indent=2) + '\n').encode(), durable)


def require_volatile_directory(path, mountinfo=Path('/proc/self/mountinfo')):
    metadata = path.lstat()
    assert stat.S_ISDIR(metadata.st_mode) and metadata.st_uid == os.getuid(), 'invalid runtime directory owner/type'
    assert stat.S_IMODE(metadata.st_mode) == 0o700, 'runtime directory must be private'
    mounts = []
    for line in mountinfo.read_text().splitlines():
        fields, filesystem = line.split(' - ', 1)
        mount = Path(fields.split()[4])
        if path == mount or path.is_relative_to(mount):
            mounts.append((len(mount.parts), filesystem.split()[0]))
    assert mounts and max(mounts)[1] == 'tmpfs', 'runtime metadata must reside on tmpfs'
