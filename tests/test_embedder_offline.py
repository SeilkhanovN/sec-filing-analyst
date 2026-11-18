"""Added in the improvement phase: the embedder loads offline once the model is cached.

sentence_transformers is replaced by a tiny fake module, so this test needs no model,
no network and no heavy import.
"""

import sys
import types

import pytest

from sec_analyst.embeddings import SentenceTransformerEmbedder


class FakeST:
    calls: list[dict] = []
    fail_offline = False

    def __init__(self, name, device=None, local_files_only=False):
        FakeST.calls.append({"name": name, "local_files_only": local_files_only})
        if local_files_only and FakeST.fail_offline:
            raise OSError("file missing from cache")


@pytest.fixture
def fake_st(monkeypatch, tmp_path):
    FakeST.calls, FakeST.fail_offline = [], False
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    return tmp_path


def test_downloads_when_not_cached(fake_st):
    emb = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    assert FakeST.calls == [{"name": "BAAI/bge-small-en-v1.5", "local_files_only": False}]
    assert not emb.loaded_offline


def test_loads_offline_when_cached(fake_st):
    (fake_st / "hub" / "models--BAAI--bge-small-en-v1.5").mkdir(parents=True)
    emb = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    assert FakeST.calls == [{"name": "BAAI/bge-small-en-v1.5", "local_files_only": True}]
    assert emb.loaded_offline


def test_falls_back_to_online_when_cache_is_incomplete(fake_st):
    (fake_st / "hub" / "models--BAAI--bge-small-en-v1.5").mkdir(parents=True)
    FakeST.fail_offline = True
    emb = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    assert [c["local_files_only"] for c in FakeST.calls] == [True, False]
    assert not emb.loaded_offline
