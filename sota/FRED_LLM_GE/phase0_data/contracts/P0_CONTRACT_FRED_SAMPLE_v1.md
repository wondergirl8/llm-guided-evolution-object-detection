# P0-CONTRACT-FRED-SAMPLE-v1

This is the frozen semantic name for the model-neutral Phase 0 sample boundary.
The Python representation lives in `phase0_data.schema.FREDSample`.

Each sample contains:

- deterministic `sample_id` in `<sequence_id>:<zero-based-frame-index>` form;
- sequence ID, frame index, and released FRED timestamp association;
- RGB and released event-frame logical references plus archive members and verified dimensions;
- zero or more unmodified `[x1, y1, x2, y2]` annotations;
- original track IDs and class strings;
- official challenging and approved project split membership;
- pinned FRED dataset/repository, schema, and manifest provenance.

The canonical manifest records the annotation/configuration policy, content hash,
scope, and validation status. A partial or smoke manifest is explicitly non-freezing.

The remote logical reference identifies the immutable source archive. Cache paths
are deliberately absent from scientific identity.

## Views

`rgb`, `event`, and `rgb_event` are views of the same sample identity. Resizing,
normalization, padding, augmentation, class collapsing, and model target conversion
belong to downstream adapters.

## Label access

- `train` may expose labels only for approved project-training samples.
- `evaluation_input` never exposes annotations.
- `trusted_evaluator` may expose labels to protected scoring/integrity code.

Callers must not infer access policy from paths or tensor counts. Changing a required
field or its meaning requires a new contract version.

`FREDDataset` is a trusted-side loader, not a security sandbox and not an object to
hand to candidate code. A protected Phase 1 evaluator may pass the returned
`evaluation_input` images/sample identity onward, but must retain the dataset object,
manifest, inventory, source adapter, and label-bearing artifacts on the trusted side.
