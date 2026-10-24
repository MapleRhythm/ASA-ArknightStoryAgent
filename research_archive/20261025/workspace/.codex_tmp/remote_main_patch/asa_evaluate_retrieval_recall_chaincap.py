#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import importlib
import sys
import types
from pathlib import Path


CHAIN_CAP = int(os.environ.get("ASA_CHAIN_TOP_N", "0"))
BASE_SCRIPT = Path("${REMOTE_HOME}/asa_evaluate_retrieval_recall_doconly_248.py")
PATCHED_SOURCE = Path("${REMOTE_HOME}/asa_chaincap_src")

if PATCHED_SOURCE.exists():
    sys.path.insert(0, str(PATCHED_SOURCE))
for module_name in list(sys.modules):
    if module_name == "goldenglow" or module_name.startswith("goldenglow."):
        del sys.modules[module_name]
importlib.invalidate_caches()

namespace = runpy.run_path(str(BASE_SCRIPT), run_name="asa_eval_base")
retriever_cls = namespace["ArknightsHybridRetriever"]
original_from_paths = retriever_cls.from_paths.__func__


def patched_from_paths(cls, *args, **kwargs):
    retriever = original_from_paths(cls, *args, **kwargs)
    original_rerank = retriever.rerank_with_evidence_chains

    def capped_rerank(instance, query, hits, **rerank_kwargs):
        del instance
        rerank_kwargs["max_chain_candidates"] = CHAIN_CAP
        return original_rerank(query, hits, **rerank_kwargs)

    retriever.rerank_with_evidence_chains = types.MethodType(capped_rerank, retriever)
    return retriever


retriever_cls.from_paths = classmethod(patched_from_paths)
sys.exit(namespace["main"]())
