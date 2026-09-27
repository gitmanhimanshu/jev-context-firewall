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
    ]

    async_tests = [
        ("API Key Pool Rotation", test_api_key_pool_rotation),
        ("Process & Prune All Three Formats", test_process_and_prune_all_three_formats),
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
