"""Atomic records and frozen-run integrity, independent of detector imports."""
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
from pathlib import Path

DEPS = ('torch', 'torchvision', 'mmdet', 'mmcv', 'mmengine', 'numpy', 'pycocotools')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha(path):
    return hashlib.sha256(json.dumps(read(path), sort_keys=True, allow_nan=False).encode()).hexdigest()


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Frozen record differs: {path}')
        return
    # Callers hold their run/task lock; records are compared structurally.
    write(path, value)


@contextlib.contextmanager
def locked(path, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def versions(python=None):
    if python:
        code = 'import json,importlib.metadata as m; print(json.dumps({n:m.version(n) for n in ' + repr(DEPS) + '}))'
        return json.loads(subprocess.check_output([str(python), '-c', code], text=True))
    return {n: importlib.metadata.version(n) for n in DEPS}


def controller_versions(python):
    code = ('import sys,json,importlib.metadata as m; print(json.dumps('
            '{"python":sys.version,"packages":{n:m.version(n) for n in '
            '("torch","deap","transformers","uvicorn","scipy")}}))')
    return json.loads(subprocess.check_output([str(python), '-c', code], text=True))


def cached_predictions(path, identity, checkpoint_sha256, predict, sample_prefix):
    path = Path(path)
    if path.exists():
        chunk = read(path)
        if chunk['identity'] != identity or chunk['checkpoint_sha256'] != checkpoint_sha256:
            raise ValueError('Sequence prediction provenance differs')
    else:
        frames, expected = predict()
        chunk = {'identity': identity, 'checkpoint_sha256': checkpoint_sha256,
                 'frames': frames, 'expected': expected}
    ids = chunk['expected']
    if (not ids or len(ids) != len(set(ids)) or [f['sample_id'] for f in chunk['frames']] != ids
            or any(not i.startswith(sample_prefix) for i in ids)):
        raise ValueError('Incomplete sequence predictions')
    if not path.exists():
        write(path, chunk)
    return chunk


def verify_run(run, detector=False):
    run = Path(run)
    manifest = read(run / 'manifest.json')
    code = run / 'code'
    for relative, expected in manifest['source_hashes'].items():
        if sha(code / relative) != expected:
            raise ValueError(f'Frozen source changed: {relative}')
    if json_sha(run / 'protocol.json') != manifest['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    for name, expected in manifest['data_hashes'].items():
        if sha(Path(name)) != expected:
            raise ValueError(f'Frozen data marker changed: {name}')
    actual = versions() if detector else versions(code / '.venv-faster-rcnn/bin/python')
    if actual != manifest['detector_versions']:
        raise ValueError('Detector dependencies changed')
    if controller_versions(code / '.venv/bin/python') != manifest['controller_versions']:
        raise ValueError('Controller/server dependencies changed')
    return manifest
