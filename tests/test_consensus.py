from jev_proxy.jev_client import ACTION_EVICT, ACTION_KEEP_FULL, arbitrate_consensus


def test_arbitrate_consensus_with_terminal_scores():
    # Chunk 13 actual evaluation from terminal:
    # noul: 0.001, topic: completely_disjoint, dep: zero_loss, waste: heavy_waste, preservation: drop_completely
    action, reason = arbitrate_consensus(
        0.001, "completely_disjoint", "zero_loss", "completed_subtask", "heavy_waste", "drop_completely", 0.60
    )
    assert action == ACTION_EVICT, f"Expected Chunk 13 to be evicted, got {action} ({reason})"

    # Chunk 8 actual evaluation:
    action8, reason8 = arbitrate_consensus(
        0.002, "completely_disjoint", "zero_loss", "completed_subtask", "heavy_waste", "drop_completely", 0.60
    )
    assert action8 == ACTION_EVICT, f"Expected Chunk 8 to be evicted, got {action8} ({reason8})"

    # Chunk 11 actual evaluation:
    action11, reason11 = arbitrate_consensus(
        0.000, "completely_disjoint", "zero_loss", "completed_subtask", "heavy_waste", "drop_completely", 0.60
    )
    assert action11 == ACTION_EVICT, f"Expected Chunk 11 to be evicted, got {action11} ({reason11})"

    # High relevance and critical loss should keep full:
    action_keep, reason_keep = arbitrate_consensus(
        0.95, "identical_thread", "critical_loss", "active_thread", "essential_tokens", "keep_full_detail", 0.60
    )
    assert action_keep == ACTION_KEEP_FULL, f"Expected keep full, got {action_keep} ({reason_keep})"
