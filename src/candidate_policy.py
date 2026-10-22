"""Optional architecture policy selected by the active problem configuration."""

from importlib import import_module


def load_candidate_policy(config):
    """Return the configured policy, or None for legacy problems."""
    module_name = config.get("CANDIDATE_POLICY_MODULE")
    return import_module(module_name) if module_name else None
