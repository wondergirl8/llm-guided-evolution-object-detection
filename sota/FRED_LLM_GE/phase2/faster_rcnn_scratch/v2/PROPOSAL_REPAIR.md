# Proposal-interface repair, November 16, 2026

The fixed-1280 run stopped at generation 0, slot 0 after 24 rejected replies.
The MMDetection baseline had completed successfully. Two integration issues were
confirmed: prompts showed only a dictionary rather than the exact class interface,
and V2 sent unformatted completion strings to the instruction-tuned Llama server.
Replies frequently used `genes` rather than `values`, contained stray/missing
Markdown fences, and included explanations. These are proposal-interface failures,
not model-training failures.

The repaired client uses the served model's local chat template with an assistant
generation header. It removes the template BOS because the tokenizer postprocessor
adds it when the server tokenizes the string. Prompts show the actual parent class,
require `values`, restrict the legal values to the requested group, and request only
one code block. Both plain and rendered prompts and the untouched raw reply are saved.

Response parsing accepts complete fenced or bare classes. It normalizes a direct
class assignment named `genes` to `values` through AST rewriting, records this
normalization, and then applies the unchanged frozen genome validator. No candidate
code is executed. Illegal values, imports, methods, extra fields, changed class
interfaces, partial dictionaries, unchanged parents, fixed-width changes, and
global duplicates remain rejected. Missing/stray fences cannot justify dropping
executable Python statements from a segment.

All 24 historical replies were replayed: 15 contain a legal changed class under
the repaired parser (including duplicate architectures); nine remain rejected.
Only the first legal globally unseen historical proposal was recovered for slot 0,
with a reference to its original reply and SHA-256. Original traces are preserved
under `run/proposals/history/proposal_repair_v1/g00_slot0/`. New replies use a new
bounded 24-attempt batch per slot. RNG state is preserved, not reset. Accepted
reply recovery runs before checking server availability.

The completed baseline, data/optimizer/resolution/sampling/epoch protocol, frozen
worker code, checkpoint identities, and selection objectives are unchanged. An
audited controller overlay under `run/control/proposal_repair/` loads the repaired
proposal function and templates while retaining the existing final_eval routing.
Its manifest pins overlay source, templates, tokenizer assets, and baseline
checkpoint identity. Automatic CPU successors use the same overlay. Do not edit
the original `run/code` or overwrite checkpoint fingerprints to resume a run.

Tests cover historical reply shapes, literal alias normalization, malformed and
executable code rejection, legal-value enforcement, no-op/global duplicate
rejection, fixed-width recovery, real class prompts, chat framing, and recovery
without an available server. The run's saved replay audit and fresh proposal
traces provide additional integration evidence beyond synthetic unit tests.

The live format repair exposed a second bottleneck: repeated legal but already
reserved class dictionaries. Revision 2 lists the exact forbidden dictionaries
for the selected class and filters out classes whose finite legal replacements
are all parents or globally seen candidates. Enumeration checks only feasibility;
it never selects or schedules a genome. The LLM still supplies every new child.
Attempt counters are preserved. The active immutable revision is pinned by
`run/control/proposal_repair_active.json`; CPU successors retain that revision.
The complete regression set passed 69 tests, and the resumed run filled all seven
initial offspring slots and submitted two training workers. The LLM server is
stopped once that proposal batch finishes.
