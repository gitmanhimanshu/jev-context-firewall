import asyncio
import sys
import traceback

from tests.test_consensus import test_arbitrate_consensus_with_terminal_scores
from tests.test_router import test_provider_router
from tests.test_keypool import test_api_key_pool_rotation, test_comma_separated_keys, test_render_format_keys
from tests.test_e2e import (
    test_atomic_turn_grouping,
    test_role_stitching,
    test_anthropic_tool_integrity,
    test_openai_tool_integrity,
    test_gemini_tool_integrity,
    test_causal_entity_linking,
    test_tail_optimizer,
    test_secret_sanitizer,
    test_process_and_prune_all_three_formats,
)
from tests.test_skeletonizer import (
    test_python_skeletonizer,
    test_go_skeletonizer,
    test_ts_skeletonizer,
    test_tail_optimizer_with_skeleton,
)
from tests.test_ollama import (
    test_ollama_evaluator_mode_config,
    test_hybrid_evaluator_mode,
)


async def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sync_tests = [
        ("Consensus Arbitration", test_arbitrate_consensus_with_terminal_scores),
        ("Provider Router", test_provider_router),
        ("Comma Separated Keys", test_comma_separated_keys),
        ("Render Format Keys", test_render_format_keys),
        ("Atomic Turn Grouping", test_atomic_turn_grouping),
        ("Role Stitching", test_role_stitching),
        ("Anthropic Tool Integrity", test_anthropic_tool_integrity),
        ("OpenAI Tool Integrity", test_openai_tool_integrity),
        ("Gemini Tool Integrity", test_gemini_tool_integrity),
        ("Causal Entity Linking", test_causal_entity_linking),
        ("Tail Optimizer", test_tail_optimizer),
        ("Secret Sanitizer", test_secret_sanitizer),
        ("Python Skeletonizer", test_python_skeletonizer),
        ("Go Skeletonizer", test_go_skeletonizer),
        ("TypeScript Skeletonizer", test_ts_skeletonizer),
        ("Tail Optimizer with Structural Skeleton", test_tail_optimizer_with_skeleton),
    ]

    async_tests = [
        ("API Key Pool Rotation", test_api_key_pool_rotation),
        ("Process & Prune All Three Formats", test_process_and_prune_all_three_formats),
        ("Ollama Evaluator Mode Config", test_ollama_evaluator_mode_config),
        ("Hybrid Offline Fail-Open Evaluator", test_hybrid_evaluator_mode),
    ]

    passed = 0
    failed = 0

    print("==================================================================")
    print("       [*] RUNNING JEV CONTEXT FIREWALL TEST SUITE (Python)       ")
    print("==================================================================")

    for name, test_fn in sync_tests:
        try:
            test_fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name} -> {e}")
            traceback.print_exc()
            failed += 1

    for name, test_fn in async_tests:
        try:
            await test_fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name} -> {e}")
            traceback.print_exc()
            failed += 1

    print("------------------------------------------------------------------")
    print(f"Results: {passed} passed, {failed} failed")
    print("==================================================================")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
