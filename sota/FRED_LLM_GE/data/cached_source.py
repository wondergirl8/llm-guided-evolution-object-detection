"""Read prepared FRED entries without provider access or cache mutations."""
from sota.FRED_LLM_GE.archive.phase0_data_old.cache import BoundedCache
from sota.FRED_LLM_GE.archive.phase0_data_old.fred_api import HFFredSource


class CachedOnlyFredSource(HFFredSource):
    def __init__(self, config, cache=None):
        # Deliberately avoid the hub client constructor.
        self.config = config
        self.cache = cache if cache is not None else BoundedCache(
            config.cache.root, config.cache.max_bytes,
            lock_timeout_seconds=config.cache.lock_timeout_seconds)
        self._active_prepared_sequences = {}

    def _with_retries(self, operation_name, operation):
        raise RuntimeError(f"provider access forbidden in cached-only mode: {operation_name}")

    def materialize_object(self, remote_object):
        raise RuntimeError("downloads and cache creation forbidden in cached-only mode")

    def list_objects(self):
        raise RuntimeError("provider listing forbidden in cached-only mode")

    def prepare_sequence(self, sequence_id, inventory_record):
        self._require_inactive_prepared_window("check cached sequence")
        with self.activate_prepared_window(((sequence_id, inventory_record),)) as prepared:
            return prepared[0]

    def cleanup_cache(self, **kwargs):
        raise RuntimeError("cache eviction forbidden in cached-only mode")
