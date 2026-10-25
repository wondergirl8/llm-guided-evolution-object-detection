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
RAW_CACHE_FORMAT = 'annotated_windows_v1'


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


def _window_bounds(events, records, window_us):
    times = events['t']
    return [(int(np.searchsorted(times, record['end_us'] - window_us, side='right')),
             int(np.searchsorted(times, record['end_us'], side='left')))
            for record in records]


def write_window_events(events, records, destination, window_us):
    """Atomically store only each annotated sample's event window.

    Offsets keep samples independent even if future protocols use overlapping
    windows.  open_memmap writes incrementally instead of copying every selected
    event into RAM a second time.
    """
    validate_events(events)
    bounds = _window_bounds(events, records, window_us)
    temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
    selected_count = sum(hi - lo for lo, hi in bounds)
    if selected_count:
        output = np.lib.format.open_memmap(temporary, mode='w+', dtype=events.dtype,
                                           shape=(selected_count,))
    else:
        with temporary.open('wb') as stream:
            np.save(stream, np.empty(0, dtype=events.dtype), allow_pickle=False)
        output = np.load(temporary, mmap_mode='r+', allow_pickle=False)
    enriched, offset = [], 0
    try:
        for record, (lo, hi) in zip(records, bounds):
            stop = offset + hi - lo
            output[offset:stop] = events[lo:hi]
            enriched.append(dict(record, event_start=offset, event_stop=stop))
            offset = stop
        output.flush()
    finally:
        del output
    temporary.replace(destination)
    return enriched


def _valid_window_index(index, source_hash, destination, window_us):
    if not (index.get('cache_format') == RAW_CACHE_FORMAT
            and index.get('source_sha256') == source_hash
            and index.get('revision') == REVISION
            and index.get('window_us') == window_us
            and destination.is_file()
            and index.get('events_sha256') == digest(destination)):
        return False
    event_count = len(np.load(destination, mmap_mode='r', allow_pickle=False))
    return all(0 <= record.get('event_start', -1) <= record.get('event_stop', -1) <= event_count
               for record in index.get('records', ()))


def prepare_sequence(source, sequence, folder, cache, window_us):
    marker = source / 'done' / f'{sequence}.json'
    annotation = json.loads(marker.read_text())
    legacy_destination = cache / f'{sequence}.npy'
    destination = cache / f'{sequence}.windows.npy'
    index_path = cache / f'{sequence}.json'
    source_hash = digest(marker)
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        if _valid_window_index(index, source_hash, destination, window_us):
            legacy_destination.unlink(missing_ok=True)
            return
        # Migrate the original full-sequence cache without another download.
        if (index.get('cache_format') is None
                and index.get('source_sha256') == source_hash
                and index.get('revision') == REVISION
                and legacy_destination.is_file()
                and index.get('events_sha256') == digest(legacy_destination)):
            events = np.load(legacy_destination, mmap_mode='r', allow_pickle=False)
            try:
                records = write_window_events(events, index['records'], destination, window_us)
            finally:
                if getattr(events, '_mmap', None) is not None:
                    events._mmap.close()
            atomic_json(index_path, dict(index, cache_format=RAW_CACHE_FORMAT,
                window_us=window_us, events_file=destination.name,
                events_sha256=digest(destination), records=records))
            legacy_destination.unlink()
            return
        else:
            raise ValueError(f'stale/corrupt raw cache {index_path}; use a fresh cache directory')
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
    records = [dict(record, end_us=times[record['frame_index']]) for record in annotation['records']]
    records = write_window_events(events, records, destination, window_us)
    atomic_json(index_path, dict(sequence=sequence, folder=folder, revision=REVISION,
        cache_format=RAW_CACHE_FORMAT, window_us=window_us, events_file=destination.name,
        source_sha256=source_hash, events_sha256=digest(destination),
        size=annotation['size'], records=records))


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
        return np.load(self.cache / f'{sequence}.windows.npy', mmap_mode='r', allow_pickle=False)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        indices = range(*index.indices(len(self))) if isinstance(index, slice) else index
        samples = []
        for i in indices:
            record = self.records[int(i)]
            events = self.events(record['sequence'])[record['event_start']:record['event_stop']]
            samples.append(voxelize(events, record['end_us'], self.settings['window_us'],
                                    self.settings['time_steps'], self.shape[-1], self.shape[-2]))
        return torch.stack(samples)


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
                    or index['folder'] != folder or index['size'] != [1280, 720]
                    or index.get('cache_format') != RAW_CACHE_FORMAT
                    or index.get('window_us') != temporal['window_us']
                    or index.get('events_file') != f'{sequence}.windows.npy'):
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
    proxy = dict(protocol['proxy'], epochs=settings['training']['epochs'],
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
    settings = protocol['families']['bn_snn']
    cache = FRED_DIR / settings['raw_cache']
    window_us = settings['events']['window_us']
    offset = 0
    for name, (source, sequences, _) in zip(('train', 'val'), search_parts(protocol)):
        manifest = {}
        for sequence, folder in sequences:
            if args.finalize:
                index_path = cache / f'{sequence}.json'
                index = json.loads(index_path.read_text())
                if (index['source_sha256'] != digest(source / 'done' / f'{sequence}.json')
                        or index.get('cache_format') != RAW_CACHE_FORMAT
                        or index.get('window_us') != window_us
                        or index.get('events_file') != f'{sequence}.windows.npy'
                        or index['events_sha256'] != digest(cache / index['events_file'])):
                    raise ValueError(f'stale/corrupt cache: {index_path}')
                manifest[sequence] = digest(index_path)
            elif offset % args.num_shards == args.shard:
                prepare_sequence(source, sequence, folder, cache, window_us)
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
