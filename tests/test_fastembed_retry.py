"""Unit tests for the fastembed model-load retry helper (CI/prod robustness).

The fastembed model download (HF/CDN) is intermittently slow or unavailable
("Could not load model ... from any source"), which repeatedly broke releases.
load_fastembed_model retries transient failures and re-raises if all fail.
"""
import sys
import types

import pytest

from hypha import utils


def test_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class _Model:
        pass

    def _ctor(model_name=None, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError(f"Could not load model {model_name} from any source.")
        return _Model()

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=_ctor))
    model = utils.load_fastembed_model("BAAI/bge-small-en-v1.5", retries=5, base_delay=0)
    assert isinstance(model, _Model)
    assert calls["n"] == 3  # failed twice, succeeded on the third attempt


def test_reraises_after_exhausting_retries(monkeypatch):
    def _ctor(model_name=None, **kwargs):
        raise ValueError(f"Could not load model {model_name} from any source.")

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=_ctor))
    with pytest.raises(ValueError, match="from any source"):
        utils.load_fastembed_model("m", retries=3, base_delay=0)


def test_passes_through_kwargs(monkeypatch):
    seen = {}

    def _ctor(model_name=None, **kwargs):
        seen["model_name"] = model_name
        seen["kwargs"] = kwargs
        return object()

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=_ctor))
    utils.load_fastembed_model("mname", retries=2, base_delay=0, cache_dir="/tmp/x")
    assert seen["model_name"] == "mname"
    assert seen["kwargs"] == {"cache_dir": "/tmp/x"}
