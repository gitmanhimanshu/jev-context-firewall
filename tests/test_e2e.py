import json

from jev_proxy.causal_graph import CausalResolver
from jev_proxy.chunker import CandidateChunk
from jev_proxy.config import Config
from jev_proxy.extractor import EntityFootprint, sanitize_secrets
from jev_proxy.jev_client import (
    ACTION_COMPRESS,
    ACTION_EVICT,
    ACTION_KEEP_FULL,
    EvaluationResult,
)
from jev_proxy.parser import (
    Message,
    group_messages_into_turns,
    stitch_messages,
)
from jev_proxy.server import ProxyServer
from jev_proxy.tail_optimizer import TailOptimizer


def test_atomic_turn_grouping():
    messages = [
        Message(role="user", content="Run tests and check errors"),
        Message(
            role="assistant",
            content="I will run tests and check config",
            tool_calls=[
                {"id": "call_1", "type": "function", "function": {"name": "run_test"}},
                {"id": "call_2", "type": "function", "function": {"name": "read_file"}},
            ],
        ),
        Message(role="tool", tool_call_id="call_1", content="Tests passed: 42"),
        Message(role="tool", tool_call_id="call_2", content="config: { port: 8080 }"),
        Message(role="assistant", content="Everything looks good."),
    ]

    _, turns = group_messages_into_turns(messages)
    assert len(turns) == 1
    assert len(turns[0].messages) == 5


def test_role_stitching():
    content_messages = [
        Message(role="user", content="First prompt"),
        Message(role="user", content="Second prompt without assistant between"),
        Message(role="assistant", content="Answer"),
    ]

    stitched = stitch_messages([], content_messages)
    assert len(stitched) == 2
    assert stitched[0].role == "user"
    assert stitched[1].role == "assistant"


def test_anthropic_tool_integrity():
    m0 = Message(role="user", content="Please read main.go")
    m1 = Message(role="assistant", content=[{"type": "text", "text": "Thinking..."}])
    m2 = Message(
        role="assistant",
        content=[
            {
                "type": "tool_use",
                "id": "call_123",
                "name": "read_file",
                "input": {"path": "main.go"},
            }
        ],
    )
    m3 = Message(
        role="user",
        content=[
            {
                "type": "tool_result",
                "tool_use_id": "call_123",
                "content": "package main",
            }
        ],
    )

    stitched = stitch_messages([], [m0, m1, m2, m3])
    assert len(stitched) == 3

    blocks = stitched[1].content
    assert len(blocks) == 2
    assert blocks[1]["type"] == "tool_use"
    assert blocks[1]["id"] == "call_123"

    # Orphaned tool_result
    orphaned_msg = Message(
        role="user",
        content=[
            {
                "type": "tool_result",
                "tool_use_id": "call_orphaned_999",
                "content": "some error",
            }
        ],
    )
    stitched_orphaned = stitch_messages([], [orphaned_msg])
    assert stitched_orphaned[0].content[0]["type"] == "text"


def test_openai_tool_integrity():
    u0 = Message(role="user", content="Fetch data")
    asst = Message(
        role="assistant",
        content="",
        tool_calls=[
            {"id": "call_A", "type": "function", "function": {"name": "func_A"}},
            {"id": "call_B", "type": "function", "function": {"name": "func_B"}},
        ],
    )
    tool_a = Message(role="tool", tool_call_id="call_A", content="output A")
    tool_b = Message(role="tool", tool_call_id="call_B", content="output B")
    user_next = Message(role="user", content="Next prompt")

    stitched = stitch_messages([], [u0, asst, tool_a, tool_b, user_next])
    assert len(stitched) == 5
    assert stitched[2].role == "tool" and stitched[2].tool_call_id == "call_A"
    assert stitched[3].role == "tool" and stitched[3].tool_call_id == "call_B"


def test_gemini_tool_integrity():
    model_msg = Message(
        role="model",
        content=[
            {
                "functionCall": {
                    "name": "calc",
                    "args": {"x": 10},
                }
            }
        ],
    )
    user_resp = Message(
        role="user",
        content=[
            {
                "functionResponse": {
                    "name": "calc",
                    "response": {"result": 20},
                }
            }
        ],
    )

    stitched = stitch_messages([], [model_msg, user_resp])
    assert len(stitched) >= 2
    last_msg = stitched[-1]
    assert "functionResponse" in last_msg.content[0]


def test_causal_entity_linking():
    chunk0 = CandidateChunk(
        id=0,
        start_turn_id=0,
        end_turn_id=5,
        entities=EntityFootprint(files=["auth/middleware.ts"], symbols=["verifyToken"]),
    )
    chunk1 = CandidateChunk(
        id=1,
        start_turn_id=6,
        end_turn_id=20,
        entities=EntityFootprint(files=["styles/main.css"]),
    )
    chunk2 = CandidateChunk(
        id=2,
        start_turn_id=21,
        end_turn_id=25,
        entities=EntityFootprint(files=["auth/middleware.ts"], errors=["403"]),
    )

    chunks = [chunk0, chunk1, chunk2]
    evaluations = [
        EvaluationResult(
            chunk_id=0,
            selected=True,
            action=ACTION_COMPRESS,
            reason="foundational_rule_compressed",
            noul=0.92,
        ),
        EvaluationResult(
            chunk_id=1,
            selected=False,
            action=ACTION_EVICT,
            reason="evicted_disjoint_zero_loss",
            noul=0.01,
        ),
        EvaluationResult(
            chunk_id=2,
            selected=True,
            action=ACTION_KEEP_FULL,
            reason="keep_full_critical",
            noul=0.98,
        ),
    ]

    resolver = CausalResolver()
    approved = resolver.resolve_dependencies(chunks, evaluations)
    approved_ids = {a.id for a in approved}

    assert 2 in approved_ids
    assert 0 in approved_ids
    assert 1 not in approved_ids


def test_tail_optimizer():
    from jev_proxy.parser import Turn
    cfg = Config()
    opt = TailOptimizer(cfg)

    turns = [
        Turn(
            id=0,
            role="assistant",
            messages=[
                Message(role="assistant", content="Viewing server.go"),
                Message(role="tool", content="package main\n\nfunc main() {\n" + ("x" * 5000) + "\n}"),
            ],
            entities=EntityFootprint(files=["server.go"]),
            token_count=1500,
        ),
        Turn(
            id=1,
            role="assistant",
            messages=[Message(role="assistant", content="I edited server.go to add port 8080")],
            entities=EntityFootprint(files=["server.go"]),
            token_count=50,
        ),
        Turn(
            id=2,
            role="user",
            messages=[Message(role="user", content="Now test it")],
            entities=EntityFootprint(),
            token_count=10,
        ),
    ]

    optimized = opt.optimize_tail(turns, "Now test it")
    assert len(optimized) == 3
    # Turn 0 tool dump should be superseded and shrunken
    turn0_tool = optimized[0].messages[1].content_string()
    assert "superseded by later edits" in turn0_tool


def test_secret_sanitizer():
    secret_text = (
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.supersecrettoken1234567890\n"
        "apiKey = 'sk-proj-99999999999999999999'\n"
        "postgres://admin:superpassword123@localhost:5432/db"
    )
    sanitized = sanitize_secrets(secret_text)
    assert "[REDACTED]" in sanitized
    assert "supersecrettoken" not in sanitized
    assert "sk-proj-99999999999999999999" not in sanitized
    assert "superpassword123" not in sanitized


async def test_process_and_prune_all_three_formats():
    cfg = Config()
    server = ProxyServer(cfg)

    # 1. Anthropic Claude format
    anthropic_json = json.dumps({
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {"role": "user", "content": "Read main.go"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Reading file..."},
                {"type": "tool_use", "id": "call_123", "name": "read_file", "input": {"path": "main.go"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_123", "content": "package main"},
            ]},
            {"role": "assistant", "content": "File read complete."},
            {"role": "user", "content": "Now add logging"},
        ],
    }).encode("utf-8")

    pruned_a, _, _, status_a = await server.process_and_prune_payload("/v1/messages", anthropic_json)
    assert status_a in ("fast_path_bypass", "pruned_success")
    res_a = json.loads(pruned_a)
    assert "messages" in res_a

    # 2. OpenAI format
    openai_json = json.dumps({
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Run tests and config"},
            {"role": "assistant", "content": "Running tools", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "run_tests", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": {"name": "read_config", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "passed"},
            {"role": "tool", "tool_call_id": "call_2", "content": '{"port": 8080}'},
            {"role": "assistant", "content": "All tests passed."},
            {"role": "user", "content": "Deploy"},
        ],
    }).encode("utf-8")

    pruned_o, _, _, status_o = await server.process_and_prune_payload("/v1/chat/completions", openai_json)
    assert status_o in ("fast_path_bypass", "pruned_success")
    res_o = json.loads(pruned_o)
    assert "messages" in res_o

    # 3. Gemini Native format
    gemini_json = json.dumps({
        "contents": [
            {"role": "user", "parts": [{"text": "Run ls command"}]},
            {"role": "model", "parts": [
                {"functionCall": {"name": "bash", "args": {"cmd": "ls"}}},
            ]},
            {"role": "user", "parts": [
                {"functionResponse": {"name": "bash", "response": {"output": "main.go"}}},
            ]},
            {"role": "model", "parts": [{"text": "Found main.go"}]},
            {"role": "user", "parts": [{"text": "Print main.go"}]},
        ],
        "systemInstruction": {
            "parts": [{"text": "You are Gemini."}],
        },
    }).encode("utf-8")

    pruned_g, _, _, status_g = await server.process_and_prune_payload(
        "/v1beta/models/gemini-1.5-pro:generateContent", gemini_json
    )
    assert status_g in ("fast_path_bypass", "pruned_success")
    res_g = json.loads(pruned_g)
    assert "contents" in res_g
