# Durable artifacts

This directory stores small, reviewable Phase 0/Phase 1 records and references.
FRED source archives, extracted sequences, runtime manifests, checkpoints, and other
large materializations remain outside Git under configured storage roots.

Phase 0's committed verification register and known-data-issue registry are seeds
for the formal audit; generated inventories, manifests, validation evidence, and
performance reports are written atomically to the configured runtime root.
