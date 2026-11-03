"""Compression receipts identify the actual version, without changing views."""
from dataclasses import replace

import pytest

from researchcoder.context.engine import EvidenceContextEngine
from researchcoder.context.policy import ContextPolicy
from tests.test_coding_context import run_tool, setup


@pytest.mark.parametrize("mode", ["off", "observed-source-v1", "observed-source-v2"])
@pytest.mark.parametrize("strategy", ["original", "evidence"])
def test_compression_statistics_match_enabled_source_contract(tmp_path, mode, strategy):
    original, agent, hooks, io = setup(tmp_path)
    policy = ContextPolicy(role="repair", strategy=strategy, allow_recovery=True,
                           coding_source_mode=mode)
    request = replace(original.request, context_policy=policy)
    engine = EvidenceContextEngine(request, io, hooks.ledger, hooks.trace)
    hooks.context_engine = engine
    hooks.bind_agent(agent)
    _, messages = run_tool(engine, agent, "read_file", {"file_path": "module.py"})
    engine.prepare(messages, agent._tool_schemas())
    stats = io.read_json(engine.pipeline.receipts[-1])["statistics"]
    if mode == "off":
        assert "coding_source_mode" not in stats
        assert "source_index_records" not in stats
        assert "source_full_manifest_sha256" not in stats
    else:
        assert stats["coding_source_mode"] == mode
        assert stats["source_index_records"] == 1
        assert stats["source_full_manifest_sha256"] == engine.coding_sources.current.digest
        assert engine.coding_sources.receipt()["mode"] == stats["coding_source_mode"]
