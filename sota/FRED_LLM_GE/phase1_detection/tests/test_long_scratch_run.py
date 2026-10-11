"""Batch loading, bounded LLM calls and interrupted-run recovery."""
import json
import pickle

import pytest
import torch
from PIL import Image

from sota.FRED_LLM_GE.phase1_detection import export_split, proxy_data


def test_lazy_training_reads_only_shuffled_batch_and_preserves_box_alignment(tmp_path, monkeypatch):
    from sota.FRED_LLM_GE.phase1_detection.tests.test_ultralytics_scratch import runner
    monkeypatch.setattr(export_split, 'sequence_frames', lambda *a: iter([
        (Image.new('RGB', (80, 40), (value, 20, 30)), index * export_split.STEP_US / 1e6)
        for index, value in ((1, 10), (2, 70), (3, 140))]))
    monkeypatch.setattr(export_split, 'load_annotations', lambda *a: {
        1: [[8., 4., 24., 20.]], 3: [[40., 4., 56., 20.]]})
    export_split.export_sequence('7', 'test', 50, tmp_path)
    data = proxy_data.load_exported_split(tmp_path, [('7', 'test')], frame_step=50)
    opened = []
    original_open = Image.open
    def counted(path):
        opened.append(path)
        return original_open(path)
    monkeypatch.setattr(Image, 'open', counted)
    batch = runner.training_batch(data, torch.tensor([2, 0]), 'cpu')
    assert [p.name for p in opened] == ['7_000003.png', '7_000001.png']
    assert batch['img'].shape == (2, 3, 64, 96)
    assert batch['batch_idx'].tolist() == [0, 1]
    torch.testing.assert_close(batch['bboxes'][:, 0], torch.tensor([48/96, 16/96]))
    assert data['images'][0:0].shape == (0, 3, 40, 80)


def test_export_rejects_stale_identity_and_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(export_split, 'sequence_frames', lambda *a: iter([
        (Image.new('RGB', (80, 40)), export_split.STEP_US / 1e6)]))
    monkeypatch.setattr(export_split, 'load_annotations', lambda *a: {})
    export_split.export_sequence('7', 'test', 50, tmp_path)
    with pytest.raises(ValueError, match='Stale'):
        export_split.export_sequence('7', 'test', 10, tmp_path)
    with pytest.raises(ValueError, match='pinned split'):
        proxy_data.load_exported_split(tmp_path, [('7', 'train')], frame_step=50)
    (tmp_path / 'images/7_000001.png').unlink()
    with pytest.raises(ValueError, match='Incomplete'):
        export_split.finalize([('7', 'test')], tmp_path, 1)


def test_resume_pairs_population_with_same_generation_global_state(tmp_path, monkeypatch):
    import run_improved as evolution
    cp, gp = tmp_path / 'cp', tmp_path / 'global'
    monkeypatch.setattr(evolution, 'GLOBAL_DATA', {'completed': {'value': 1}})
    monkeypatch.setattr(evolution, 'GLOBAL_DATA_HIST', {})
    monkeypatch.setattr(evolution, 'GLOBAL_DATA_ANCESTRY', {})
    evolution.save_checkpoint(3, str(cp), str(gp), {'population': ['old'], 'hof': []})
    # Interruption between global and population writes leaves an orphan global.
    evolution.atomic_pickle(str(gp / 'global_gen_4.pkl'), {'GLOBAL_DATA': {'orphan': True}})
    pop, next_gen, state = evolution.load_checkpoint(str(cp), global_path=str(gp))
    assert pop['population'] == ['old'] and next_gen == 4
    assert state['GLOBAL_DATA'] == {'completed': {'value': 1}}
    assert not list(cp.glob('*.tmp'))
    (gp / 'global_gen_3.pkl').unlink()
    with pytest.raises(FileNotFoundError, match='matching global'):
        evolution.load_checkpoint(str(cp), global_path=str(gp))


def test_failed_checkpoint_write_preserves_previous_state(tmp_path, monkeypatch):
    import run_improved as evolution
    path = str(tmp_path / 'checkpoint.pkl')
    evolution.atomic_pickle(path, {'completed': 3})
    def killed(data, file):
        file.write(b'partial')
        raise OSError('interrupted')
    monkeypatch.setattr(evolution.pickle, 'dump', killed)
    with pytest.raises(OSError):
        evolution.atomic_pickle(path, {'completed': 4})
    with open(path, 'rb') as file:
        assert pickle.load(file) == {'completed': 3}
    assert not list(tmp_path.glob('*.tmp'))


def test_llm_client_uses_configured_tokens_timeout_and_explicit_url(monkeypatch):
    from src import llm_utils
    calls = []
    class Response:
        status_code = 200
        def json(self):
            return {'generated_text': 'candidate', 'response_time_sec': 1}
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()
    monkeypatch.setattr(llm_utils.requests, 'post', post)
    monkeypatch.setattr(llm_utils, 'LLM_MAX_NEW_TOKENS', 1648)
    monkeypatch.setenv('LLMGE_LLM_REQUEST_TIMEOUT_SECONDS', '600')
    assert llm_utils.submit_mixtral_local('prompt', server_url='http://fake/generate') == 'candidate'
    assert calls[0][0] == 'http://fake/generate'
    assert calls[0][1]['json']['max_new_tokens'] == 1648
    assert calls[0][1]['timeout'] == (10, 600)
    llm_utils.submit_mixtral_local('prompt', max_new_tokens=200, server_url='http://fake/generate')
    assert calls[1][1]['json']['max_new_tokens'] == 200


def test_missing_server_stops_option_b_controller_before_more_jobs(monkeypatch):
    import run_improved as evolution
    monkeypatch.setenv('FRED_DATA_PROFILE', 'b')
    monkeypatch.setattr(evolution, 'wait_for_llm_server', lambda: False)
    with pytest.raises(RuntimeError, match='restart the server and resume'):
        evolution.submit_bash('unused')


def test_whole_zip_export_uses_same_timestamp_sampling_for_later_names(tmp_path):
    import io
    import zipfile
    from sota.FRED_LLM_GE.data.fred_stream import zip_event_frames
    image = io.BytesIO()
    Image.new('RGB', (8, 4)).save(image, format='PNG')
    archive = tmp_path / '116.zip'
    with zipfile.ZipFile(archive, 'w') as file:
        for index in range(1, 52):
            file.writestr(f'116/Event/Frames/Video_116_{index * export_split.STEP_US}.png', image.getvalue())
    frames = list(zip_event_frames(archive, 'train', 116, tmp_path / 'report.csv', 50))
    assert [round(t * 1e6) for _, t in frames] == [export_split.STEP_US, 51 * export_split.STEP_US]
    assert all(image.size == (8, 4) for image, _ in frames)
