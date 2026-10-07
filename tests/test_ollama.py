from jev_proxy.config import Config
from jev_proxy.jev_client import ACTION_KEEP_FULL, JevClient


async def test_ollama_evaluator_mode_config():
    cfg = Config(
        evaluator_mode="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen2.5-coder:1.5b",
    )
    client = JevClient(cfg)
    assert client.cfg.evaluator_mode == "ollama"

    # Testing offline fallback: when Ollama is unreachable on test port, it must fail-open gracefully
    res = await client.evaluate_batch(
        chunk_id=1,
        state={"current_query": "test query", "chunk_summary": []},
    )
    assert res.selected is True
    assert res.action == ACTION_KEEP_FULL
    assert "fail_open" in res.reason


async def test_hybrid_evaluator_mode():
    cfg = Config(
        evaluator_mode="hybrid",
        jev_base_url="https://invalid-non-existent-domain-1234.ai",
        ollama_base_url="http://localhost:59999",  # unrouted port
    )
    client = JevClient(cfg)
    res = await client.evaluate_batch(
        chunk_id=2,
        state={"current_query": "hybrid test", "chunk_summary": []},
    )
    # Must fail open safely without raising unhandled exception
    assert res.selected is True
    assert res.action == ACTION_KEEP_FULL

