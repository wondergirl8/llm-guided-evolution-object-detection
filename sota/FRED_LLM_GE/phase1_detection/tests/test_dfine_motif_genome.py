"""Focused checks for the opt-in, bounded D-FINE architecture search."""

import ast
import json
from pathlib import Path
import random

import pytest

from src import fred_dfine_genome as motifs


SEED = Path(__file__).parents[2] / "seeds/seed_dfine.py"
TEMPLATE = "Choose one value for this genome: {}\nReturn JSON only."


def test_motif_catalog_renders_196_distinct_valid_architectures():
    hashes = set()
    for backbone in motifs.CHOICES["backbone"]:
        for encoder in motifs.CHOICES["encoder"]:
            for decoder in motifs.CHOICES["decoder"]:
                genome = {"version": motifs.VERSION, "genes": {
                    "backbone": backbone, "encoder": encoder, "decoder": decoder}}
                source = motifs.render_source(SEED, genome)
                assert source.count("# --OPTION--") == 3
                hashes.add(ast.dump(ast.parse(source), include_attributes=False))
    assert len(hashes) == 196
    assert motifs.render_source(SEED, motifs.seed_genome()) == SEED.read_text()


def test_initial_coverage_is_audited_and_crossover_recombines_genes(tmp_path):
    choices = {"backbone": "p4_dw_k3_s025",
               "encoder": "bidir_td050_bu025",
               "decoder": "p4_quota_50"}
    entries = []
    for gene, value in choices.items():
        output = tmp_path / f"network_{gene}.py"
        motifs.generate_mutation(
            SEED, output, SEED, TEMPLATE, lambda prompt, value=value: json.dumps({"value": value}),
            coverage_target=gene)
        assert motifs.load_genome(output, SEED)["genes"][gene] == value
        entries.append((gene, output))
    report = motifs.verify_initial_coverage(entries, SEED)
    assert [item["target"] for item in report["candidates"]] == list(motifs.GENES)

    child = tmp_path / "network_combined.py"
    motifs.generate_crossover(entries[0][1], entries[1][1], child, SEED, random.Random(0))
    assert motifs.load_genome(child, SEED)["genes"] == {
        "backbone": choices["backbone"], "encoder": choices["encoder"], "decoder": "none"}


def test_motif_response_and_source_identity_are_strict(tmp_path):
    output = tmp_path / "network_bad.py"
    with pytest.raises(RuntimeError, match="No valid D-FINE"):
        motifs.generate_mutation(SEED, output, SEED, TEMPLATE,
                                 lambda prompt: '{"value":"none"}',
                                 coverage_target="backbone", attempts=2)
    assert not output.exists()
    audit = json.loads(Path(str(output) + ".proposal_attempts.json").read_text())
    assert len(audit["attempts"]) == 2

    good = tmp_path / "network_good.py"
    motifs.generate_mutation(SEED, good, SEED, TEMPLATE,
                             lambda prompt: '{"value":"p4_dw_k3_s050"}',
                             coverage_target="backbone")
    good.write_text(good.read_text() + "\n# unauthorized edit\n")
    with pytest.raises(ValueError, match="source and"):
        motifs.load_genome(good, SEED)


def test_motif_uses_first_complete_json_answer_from_continuing_llm(tmp_path):
    # PACE's local Llama server continued after a valid fenced answer with
    # tool-call-like text and further choices, exhausting all five attempts.
    raw = ('```json\n{"value": "p4_dw_k3_s025"}\n```\n\n'
           '{"type": "function", "name": "choose_next_value", "parameters": {}}'
           '```json\n{"value": "p4_dw_k5_s100"}\n```')
    output = tmp_path / 'network_first.py'
    motifs.generate_mutation(SEED, output, SEED, TEMPLATE,
                             lambda _: raw, coverage_target='backbone')
    assert motifs.load_genome(output, SEED)['genes']['backbone'] == 'p4_dw_k3_s025'
    audit = json.loads(Path(str(output) + '.proposal_attempts.json').read_text())
    assert audit['attempts'][0]['proposal'] == raw
    assert audit['attempts'][0]['accepted'] is True


def test_repeated_valid_duplicate_uses_audited_bounded_fallback(tmp_path):
    first = tmp_path / 'network_first.py'
    second = tmp_path / 'network_second.py'
    reply = '{"value":"bidir_td050_bu025"}'
    motifs.generate_mutation(SEED, first, SEED, TEMPLATE, lambda _: reply,
                             coverage_target='encoder')
    prompts = []

    def repeat(prompt):
        prompts.append(prompt)
        return reply

    motifs.generate_mutation(SEED, second, SEED, TEMPLATE, repeat,
                             coverage_target='encoder')
    assert len(prompts) == 5
    assert 'bidir_td050_bu025' not in json.loads(
        prompts[1].split('genome: ', 1)[1].split('\n', 1)[0])['allowed_values']
    assert motifs.load_genome(second, SEED)['genes']['encoder'] == 'bidir_td050_bu050'
    audit = json.loads(Path(str(second) + '.proposal_attempts.json').read_text())
    assert len(audit['attempts']) == 6
    assert all('Architecture duplicates' in item['error'] for item in audit['attempts'][:5])
    assert audit['attempts'][5]['accepted'] is True
    assert audit['attempts'][5]['selection_source'] == 'bounded_duplicate_fallback'
    assert audit['attempts'][5]['selected_value'] == 'bidir_td050_bu050'


def test_motif_does_not_skip_invalid_first_answer():
    assert motifs._proposal_value(
        '{"value":"p4_dw_k3_s025"}\nextra generated text', 'backbone'
    ) == 'p4_dw_k3_s025'
    raw = ('```json\n{"value": "not_allowed"}\n```\n'
           '```json\n{"value": "p4_dw_k3_s025"}\n```')
    with pytest.raises(ValueError, match='invalid backbone value'):
        motifs._proposal_value(raw, 'backbone')
    with pytest.raises(ValueError, match='invalid JSON fence'):
        motifs._proposal_value('```python\n{"value":"p4_dw_k3_s025"}\n```', 'backbone')


def test_motif_accepts_unambiguous_live_llama_answer_forms():
    current = motifs.seed_genome()["genes"]
    assert motifs._proposal_value(
        '```json\n{"gene":"encoder","value":"bidir_td050_bu025"}\n```',
        'encoder', current) == 'bidir_td050_bu025'
    assert motifs._proposal_value(
        '```json\n{"backbone":"p4_dw_k3_s025","encoder":"none","decoder":"none"}\n```',
        'backbone', current) == 'p4_dw_k3_s025'
    assert motifs._proposal_value(
        'I considered the allowed values.\nThe final answer is: '
        '$\\boxed{p4_quota_50}$', 'decoder', current) == 'p4_quota_50'


def test_motif_rejects_ambiguous_or_wrong_gene_answers():
    current = motifs.seed_genome()["genes"]
    replies = [
        ('{"gene":"encoder","value":"p4_dw_k3_s025"}', 'backbone'),
        ('{"backbone":"p4_dw_k3_s025","encoder":"bidir_td050_bu025",'
         '"decoder":"none"}', 'backbone'),
        ('{"type":"function","name":"choose_value","parameters":{}}', 'encoder'),
        ('The final answer is: $\\boxed{p4_quota_50}$\n'
         'Another answer: $\\boxed{p4_quota_75}$', 'decoder'),
        ('Choose p4_quota_50', 'decoder'),
    ]
    for reply, target in replies:
        with pytest.raises(ValueError):
            motifs._proposal_value(reply, target, current)


def test_dfine_motifs_local_prompt_uses_instruct_chat_template(monkeypatch):
    from src import llm_utils

    class Tokenizer:
        bos_token = '<bos>'

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert messages == [{'role': 'user', 'content': 'choose a motif'}]
            assert tokenize is False and add_generation_prompt is True
            return '<bos><user>choose a motif</user><assistant>'

    monkeypatch.setattr(llm_utils, 'SEED_FAMILY', 'dfine')
    monkeypatch.setattr(llm_utils, 'DFINE_SEARCH_SPACE', 'motifs_v1')
    monkeypatch.setattr(llm_utils, '_local_chat_tokenizer', lambda _: Tokenizer())
    assert llm_utils._format_local_prompt('choose a motif') == (
        '<user>choose a motif</user><assistant>')


def test_opt_in_mutation_routes_json_through_the_structured_builder(tmp_path, monkeypatch):
    from src import llm_mutation

    monkeypatch.setattr(llm_mutation, "SEED_FAMILY", "dfine")
    monkeypatch.setattr(llm_mutation, "SEED_NETWORK", str(SEED))
    prompts = []

    def reply(prompt, *args, **kwargs):
        prompts.append((prompt, kwargs))
        return '{"value":"p4_quota_25"}'

    monkeypatch.setattr(llm_mutation, "generate_augmented_code", reply)
    template = tmp_path / "prompt.txt"
    template.write_text(TEMPLATE)
    output = tmp_path / "network_1.py"
    llm_mutation.augment_network(str(SEED), str(output), str(template),
                                 coverage_target="decoder", dfine_search_space="motifs_v1")
    assert motifs.load_genome(output, SEED)["genes"]["decoder"] == "p4_quota_25"
    assert len(prompts) == 1 and prompts[0][1]["return_raw"] is True


def test_controller_assigns_one_initial_slot_per_motif(monkeypatch):
    import run_improved as controller

    monkeypatch.setattr(controller, "SEED_FAMILY", "dfine")
    monkeypatch.setattr(controller, "DFINE_SEARCH_SPACE", "motifs_v1")
    monkeypatch.setattr(controller.toolbox, "individual",
                        lambda **kwargs: kwargs.get("coverage_target"))
    panel = controller.create_population(7, "test-model", coverage_panel=True)
    assert panel == ["backbone", "encoder", "decoder", None, None, None, None]


def test_evaluator_rejects_candidate_missing_structured_genome(tmp_path, monkeypatch):
    from sota.FRED_LLM_GE.phase1_detection import candidate_runner

    candidate = tmp_path / "network_bad.py"
    candidate.write_text(SEED.read_text() + "\n# legacy edit\n")

    class FakeAdapter:
        def __init__(self, settings, fred_dir):
            self.settings = settings

        def variant_path(self, model_name, variant_dir):
            return candidate

        def build(self, variant_path, num_classes, device):
            return object()

        def param_count(self, model):
            return 1

    monkeypatch.setitem(candidate_runner.ADAPTERS, "dfine", FakeAdapter)
    failures = []
    monkeypatch.setattr(candidate_runner, "_record_run_failure",
                        lambda *args: failures.append(args[-1]))
    candidate_runner.run("bad", "network_bad", tmp_path, "dfine",
                         {"families": {"dfine": {}}, "proxy": {"seed": 0}},
                         tmp_path, "cpu", dfine_search_space="motifs_v1")
    assert len(failures) == 1
    assert "genome" in str(failures[0])


def test_resource_screen_keeps_weights_and_batchnorm_state():
    from sota.FRED_LLM_GE.phase1_detection.adapters.dfine import DFINEAdapter
    import yaml

    fred_dir = SEED.parent.parent
    protocol = yaml.safe_load((fred_dir / "configs/phase1/protocol.yaml").read_text())
    adapter = DFINEAdapter(protocol["families"]["dfine"], fred_dir)
    model = adapter.build(SEED, 1, "cpu")
    before = {key: value.clone() for key, value in model.state_dict().items()}
    probe = adapter.screen_resources(model, (65, 97), 1, "cpu")
    assert probe["full_resolution_probe_shape"] == [65, 97]
    assert probe["full_resolution_probe_peak_bytes"] is None
    assert model.training
    assert all((value == before[key]).all() for key, value in model.state_dict().items())
    adapter.settings["search_resource_limits"] = {"max_params": 1}
    with pytest.raises(ValueError, match="parameters"):
        adapter.screen_resources(model, (65, 97), 1, "cpu")
