"""Prepare and lazily bin raw FRED events at the existing search annotation times.

Run as a module with --shard N --num-shards K, then --finalize. Preparation
reads official challenging-train sequences only; existing PNG export done
markers supply the already-converted labels. PNG pixels are never loaded.
"""
import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml

from .export_split import dataset_parts
from .fred_splits import downloaded_sequence
from .proxy_data import REVISION, STEP_US
from fred_events import load_events
from fred_stream import _select_frames, _write_report

FRED_DIR = Path(__file__).resolve().parents[1]
PROTOCOL = FRED_DIR / 'configs/phase1/protocol.yaml'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True) + '\n')
    temporary.replace(path)


def search_parts(protocol):
    if protocol['proxy']['official_split'] != 'challenging_train':
        raise ValueError('BN-SNN search must use challenging_train')
    # Use released-frame annotations, independently of optional CNN event sweeps.
    challenging = dict(protocol['challenging'], events={'source': 'released'})
    return dataset_parts(challenging, FRED_DIR / challenging['data_root'], 'search')


def prepare_sequence(source, sequence, folder, cache):
    marker = source / 'done' / f'{sequence}.json'
    annotation = json.loads(marker.read_text())
    destination = cache / f'{sequence}.npy'
    index_path = cache / f'{sequence}.json'
    source_hash = digest(marker)
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        if (index['source_sha256'] != source_hash or index['revision'] != REVISION
                or index['events_sha256'] != digest(destination)):
            raise ValueError(f'stale/corrupt raw cache {index_path}; use a fresh cache directory')
        return
    if annotation['sequence'] != sequence or annotation['hf_folder'] != folder:
        raise ValueError(f'annotation identity mismatch: {marker}')
    if annotation['size'] != [1280, 720]:
        raise ValueError(f'unexpected FRED sensor dimensions: {marker}')
    cache.mkdir(parents=True, exist_ok=True)
    with downloaded_sequence(folder, sequence, cache / 'downloads') as archive_path:
        with zipfile.ZipFile(archive_path) as archive:
            frames, bad = _select_frames(int(sequence), archive.namelist())
            _write_report(cache / f'bad_timestamps_{sequence}.csv', folder, sequence, bad)
            times = {}
            for timestamp, _ in frames:
                frame = round(timestamp / STEP_US)
                if frame in times:
                    raise ValueError(f'ambiguous frame time: {sequence}:{frame}')
                times[frame] = timestamp
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                events = load_events(archive.extract(f'{sequence}/Event/events.hdf5', temporary))
            validate_events(events)
            temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
            with temporary.open('wb') as stream:
                np.save(stream, events, allow_pickle=False)
            temporary.replace(destination)
    records = [dict(record, end_us=times[record['frame_index']]) for record in annotation['records']]
    atomic_json(index_path, dict(sequence=sequence, folder=folder, revision=REVISION,
        source_sha256=source_hash, events_sha256=digest(destination), size=annotation['size'], records=records))


def validate_events(events):
    if events.ndim != 1 or events.dtype.names is None or not {'x', 'y', 't', 'p'} <= set(events.dtype.names):
        raise ValueError('raw events require a 1-D structured array with x,y,t,p fields')
    if any(events.dtype.fields[k][0].kind not in 'iu' for k in ('x', 'y', 't', 'p')):
        raise ValueError('raw event fields must be integer-valued')
    if (np.any(np.diff(events['t'].astype(np.int64)) < 0)
            or np.any(events['x'] >= 1280) or np.any(events['y'] >= 720)
            or np.any(events['x'] < 0) or np.any(events['y'] < 0)
            or not np.isin(events['p'], (0, 1)).all()):
        raise ValueError('raw event times/coordinates/polarity violate FRED contract')


def voxelize(events, end_us, window_us, time_steps, width, height):
    """Count events in fixed physical bins of the open window (end-window,end).

    No per-sample stretching to the first/last observed event; silent bins stay
    silent. Channels 0/1 are negative/positive. Spatial resolution is native.
    """
    if window_us <= 0 or time_steps < 2:
        raise ValueError('window must be positive and time_steps at least two')
    times = events['t']
    start = end_us - window_us
    lo, hi = np.searchsorted(times, start, side='right'), np.searchsorted(times, end_us, side='left')
    selected = events[lo:hi]
    bins = ((selected['t'].astype(np.int64) - start) * time_steps // window_us).astype(np.int64)
    flat = ((bins * 2 + selected['p'].astype(np.int64)) * height + selected['y']) * width + selected['x']
    counts = np.bincount(flat, minlength=time_steps * 2 * height * width)
    return torch.from_numpy(counts.reshape(time_steps, 2, height, width).astype(np.float32))


class LazyEvents:
    """The image-batch interface of candidate_runner, extended to [B,T,2,H,W]."""
    def __init__(self, records, cache, settings, size):
        self.records, self.cache, self.settings = records, Path(cache), settings
        width, height = size
        self.shape = (len(records), settings['time_steps'], 2, height, width)
        # Per-instance bounded mmap cache; no entire sequence copied into RAM.
        self.events = lru_cache(maxsize=2)(self._open)

    def _open(self, sequence):
        return np.load(self.cache / f'{sequence}.npy', mmap_mode='r', allow_pickle=False)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        indices = range(*index.indices(len(self))) if isinstance(index, slice) else index
        return torch.stack([voxelize(self.events(self.records[int(i)]['sequence']),
            self.records[int(i)]['end_us'], self.settings['window_us'], self.settings['time_steps'],
            self.shape[-1], self.shape[-2]) for i in indices])


def load_search(protocol):
    settings = protocol['families']['bn_snn']
    temporal = settings['events']
    if (temporal['representation'] != 'count' or temporal['time_steps'] < 2
            or temporal['time_steps'] % 2 or temporal['window_us'] <= 0):
        raise ValueError('BN-SNN requires count bins, an even time_steps >=2, and positive window_us')
    cache = FRED_DIR / settings['raw_cache']
    parts = search_parts(protocol)
    datasets = []
    for name, (source, sequences, step) in zip(('train', 'val'), parts):
        manifest = json.loads((cache / f'{name}_manifest.json').read_text())
        expected = {}
        records = []
        for sequence, folder in sequences:
            index_path = cache / f'{sequence}.json'
            index = json.loads(index_path.read_text())
            if (index['source_sha256'] != digest(source / 'done' / f'{sequence}.json')
                    or index['revision'] != REVISION or index['sequence'] != sequence
                    or index['folder'] != folder or index['size'] != [1280, 720]):
                raise ValueError(f'stale raw event index: {index_path}')
            expected[sequence] = digest(index_path)
            records.extend(dict(r, sequence=sequence, folder=folder) for r in index['records'])
        if manifest != expected or not records:
            raise ValueError(f'BN-SNN {name} manifest is stale/empty; rerun preparation --finalize')
        boxes = [torch.tensor(r['boxes'], dtype=torch.float32).reshape(-1, 4) for r in records]
        datasets.append(dict(images=LazyEvents(records, cache, temporal, (1280, 720)),
            boxes=boxes, orig_boxes=boxes, orig_size=(1280, 720),
            sample_ids=[f"{r['folder']}/{r['sequence']}:{r['frame_index']}" for r in records],
            manifest_sha256=digest(cache / f'{name}_manifest.json')))
    proxy = dict(protocol['proxy'], epochs=protocol['challenging']['search']['epochs'],
        train_sequences=[s for s, _ in parts[0][1]], val_sequences=[s for s, _ in parts[1][1]],
        frame_step=parts[0][2], val_frame_step=parts[1][2])
    if proxy['img_width'] != 1280:
        raise ValueError('BN-SNN initial protocol uses native 1280x720; resizing is not implemented')
    return proxy, datasets[0], lambda: datasets[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error('require 0 <= shard < num-shards')
    protocol = yaml.safe_load(args.protocol.read_text())
    cache = FRED_DIR / protocol['families']['bn_snn']['raw_cache']
    offset = 0
    for name, (source, sequences, _) in zip(('train', 'val'), search_parts(protocol)):
        manifest = {}
        for sequence, folder in sequences:
            if args.finalize:
                index_path = cache / f'{sequence}.json'
                index = json.loads(index_path.read_text())
                if (index['source_sha256'] != digest(source / 'done' / f'{sequence}.json')
                        or index['events_sha256'] != digest(cache / f'{sequence}.npy')):
                    raise ValueError(f'stale/corrupt cache: {index_path}')
                manifest[sequence] = digest(index_path)
            elif offset % args.num_shards == args.shard:
                prepare_sequence(source, sequence, folder, cache)
                print(f'Prepared {folder}/{sequence}', flush=True)
            offset += 1
        if args.finalize:
            atomic_json(cache / f'{name}_manifest.json', manifest)
            print(f'Finalized {name}: {len(manifest)} sequences')
    if args.finalize:
        _, train, val = load_search(protocol)
        print(f'Windows: train={len(train["images"])} val={len(val()["images"])}')


if __name__ == '__main__':
    main()
