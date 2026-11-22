"""Focused isolation and request-integrity checks for interim testing."""
from unittest.mock import Mock

import pytest
from sota.FRED_LLM_GE.phase2.faster_rcnn_scratch import interim_eval as evaluation


@pytest.mark.parametrize('change', [
    {'completed_epochs': 9}, {'purpose': 'final'}, {'use_for_selection': True},
])
def test_rejects_wrong_epoch_or_selection_use(tmp_path, monkeypatch, change):
    request = dict(completed_epochs=8, purpose='interim_official_test',
                   use_for_selection=False, **{})
    request.update(change)
    monkeypatch.setattr(evaluation, 'load_json', lambda _: request)
    with pytest.raises(ValueError, match='Invalid interim'):
        evaluation.verify(tmp_path, tmp_path)


def test_rejects_changed_evaluator(tmp_path, monkeypatch):
    request = dict(completed_epochs=8, purpose='interim_official_test',
                   use_for_selection=False, runner_sha256='original')
    monkeypatch.setattr(evaluation, 'load_json', lambda _: request)
    monkeypatch.setattr(evaluation, 'file_sha256', lambda _: 'changed')
    with pytest.raises(ValueError, match='runner changed'):
        evaluation.verify(tmp_path, tmp_path)


def test_verify_only_does_not_build_or_use_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, 'verify', lambda *_: (None,) * 7)
    build = Mock(side_effect=AssertionError('Must not build during verification'))
    monkeypatch.setattr(evaluation, 'FasterRCNNScratchAdapter', build)
    monkeypatch.setattr(evaluation.torch.cuda, 'is_available',
                        Mock(side_effect=AssertionError('Must not require GPU')))
    evaluation.evaluate(tmp_path, tmp_path, verify_only=True)
    build.assert_not_called()


def test_no_silent_cpu_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, 'verify', lambda *_: (None,) * 7)
    monkeypatch.setattr(evaluation.torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='requires a CUDA'):
        evaluation.evaluate(tmp_path, tmp_path)
