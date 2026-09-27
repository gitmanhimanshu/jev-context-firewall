import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from jev_proxy.config import Config
from jev_proxy.extractor import EntityFootprint, has_anaphora, sanitize_secrets
from jev_proxy.parser import (
    Message,
    Turn,
    estimate_tokens,
    extract_real_user_query,
    has_tool_use,
    is_tool_result_message,
)


@dataclass
class CandidateChunk:
    id: int
    start_turn_id: int
    end_turn_id: int
    turns: List[Turn] = field(default_factory=list, repr=False)
    total_tokens: int = 0
    entities: EntityFootprint = field(default_factory=EntityFootprint)
    summary: Any = None
    is_lightweight: bool = False
    is_foundational: bool = False
    has_tools: bool = False


class Chunker:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def partition(self, turns: List[Turn], current_query: str) -> Tuple[List[Turn], List[CandidateChunk], str]:
        session_map = build_session_map(turns)

        if len(turns) <= 1:
            return turns, [], session_map

        # Always keep the active turn (most recent turn) sacrosanct
        tail_size = 1

        # If query has anaphora ("undo that", "why did it fail"), preserve the immediate previous turn too
        if has_anaphora(current_query) and len(turns) >= 2:
            tail_size = 2

        split_idx = len(turns) - tail_size
        historical_turns = turns[:split_idx]
        tail_turns = turns[split_idx:]

        historical_tokens = sum(t.token_count for t in historical_turns)

        # If previous history is very lightweight (< 600 tokens), bypass Jev for latency
        if historical_tokens < 600:
            return turns, [], session_map

        max_tokens = self.cfg.chunk_max_tokens if self.cfg.chunk_max_tokens > 0 else 1000

        # Decompose historical turns into atomic sub-turn units
        atomic_units: List[Turn] = []
        for t in historical_turns:
            atomic_units.extend(decompose_turn(t, max_tokens))

        chunks: List[CandidateChunk] = []
        current_batch: List[Turn] = []
        current_tokens = 0

        for i, unit in enumerate(atomic_units):
            if current_tokens + unit.token_count > max_tokens and current_batch:
                chunks.append(create_candidate_chunk(len(chunks), current_batch, current_tokens))
                current_batch = []
                current_tokens = 0

            current_batch.append(unit)
            current_tokens += unit.token_count

            if i == len(atomic_units) - 1 and current_batch:
                chunks.append(create_candidate_chunk(len(chunks), current_batch, current_tokens))
                current_batch = []
                current_tokens = 0

        return tail_turns, chunks, session_map


def slice_text_by_tokens(text: str, max_tokens: int) -> List[str]:
    if not text:
        return []
    max_chars = int(max_tokens * 2.36)
    if max_chars < 500:
        max_chars = 500

    if len(text) <= max_chars:
        return [text]

    lines = text.split("\n")
    slices: List[str] = []
    cur_lines: List[str] = []
    cur_len = 0

    for line in lines:
        cur_lines.append(line)
        cur_len += len(line) + 1
        if cur_len >= max_chars:
            slices.append("\n".join(cur_lines))
            cur_lines = []
            cur_len = 0

    if cur_lines:
        slices.append("\n".join(cur_lines))

    return slices


def slice_message_by_tokens(m: Message, max_tokens: int) -> List[Message]:
    tok = estimate_tokens(m.content_string())
    if tok <= max_tokens:
        return [m]

    # 1. Anthropic structured blocks or Gemini parts
    if isinstance(m.content, list):
        result_msgs: List[Message] = []
        for b in m.content:
            if isinstance(b, dict):
                # Anthropic tool_result
                if b.get("type") == "tool_result":
                    tool_id = b.get("tool_use_id", "")
                    content_val = b.get("content", "")
                    content_str = content_val if isinstance(content_val, str) else json.dumps(content_val)

                    text_slices = slice_text_by_tokens(content_str, max_tokens)
                    for s in text_slices:
                        result_msgs.append(
                            Message(
                                role=m.role,
                                content=[
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_id,
                                        "content": s,
                                    }
                                ],
                            )
                        )
                    continue

                # Gemini functionResponse
                if "functionResponse" in b and isinstance(b["functionResponse"], dict):
                    fr = b["functionResponse"]
                    name = fr.get("name", "")
                    raw_resp = json.dumps(fr.get("response", {}))
                    text_slices = slice_text_by_tokens(raw_resp, max_tokens)
                    for s in text_slices:
                        result_msgs.append(
                            Message(
                                role=m.role,
                                content=[
                                    {
                                        "functionResponse": {
                                            "name": name,
                                            "response": {"result": s},
                                        }
                                    }
                                ],
                            )
                        )
                    continue

                # Text block
                if isinstance(b.get("text"), str) and estimate_tokens(b["text"]) > max_tokens:
                    text_slices = slice_text_by_tokens(b["text"], max_tokens)
                    for s in text_slices:
                        block = {"text": s}
                        if "type" in b:
                            block["type"] = "text"
                        result_msgs.append(Message(role=m.role, content=[block]))
                    continue

            # Under max_tokens block
            result_msgs.append(Message(role=m.role, content=[b]))

        if result_msgs:
            return result_msgs

    # 2. OpenAI tool message
    if m.role in ("tool", "function") or m.tool_call_id:
        text_slices = slice_text_by_tokens(m.content_string(), max_tokens)
        return [
            Message(role=m.role, tool_call_id=m.tool_call_id, content=s)
            for s in text_slices
        ]

    # 3. Plain text message
    text_slices = slice_text_by_tokens(m.content_string(), max_tokens)
    return [
        Message(role=m.role, content=s, tool_calls=m.tool_calls)
        for s in text_slices
    ]


def decompose_turn(turn: Turn, max_tokens: int) -> List[Turn]:
    messages = turn.messages
    if not messages:
        return []

    sliced_messages: List[Message] = []
    for m in messages:
        sliced_messages.extend(slice_message_by_tokens(m, max_tokens))

    sub_turns: List[Turn] = []
    i = 0
    n = len(sliced_messages)

    while i < n:
        m = sliced_messages[i]
        unit = [m]

        if (
            i == 0
            and m.role == "user"
            and i + 1 < n
            and sliced_messages[i + 1].role in ("assistant", "model")
            and has_tool_use(sliced_messages[i + 1])
        ):
            i += 1
            m = sliced_messages[i]
            unit.append(m)

        if m.role in ("assistant", "model") and has_tool_use(m):
            if i + 1 < n and is_tool_result_message(sliced_messages[i + 1]):
                i += 1
                unit.append(sliced_messages[i])

        i += 1

        unit_text = " ".join(um.content_string() for um in unit)
        tok = sum(estimate_tokens(um.content_string()) for um in unit)
        sub_ent = EntityFootprint()
        from jev_proxy.extractor import extract_entities
        sub_ent = extract_entities(unit_text)
        if not sub_ent.files and turn.entities.files:
            sub_ent.files = list(turn.entities.files)

        sub_turns.append(
            Turn(
                id=turn.id,
                role=turn.role,
                messages=unit,
                entities=sub_ent,
                token_count=tok,
            )
        )

    return sub_turns


def create_candidate_chunk(chunk_id: int, turns: List[Turn], tokens: int) -> CandidateChunk:
    has_tools = any(turn_has_tools(t) for t in turns)
    is_foundational = chunk_id == 0 and not has_tools
    is_lightweight = not has_tools and tokens < 300 and len(turns) <= 2

    return CandidateChunk(
        id=chunk_id,
        start_turn_id=turns[0].id,
        end_turn_id=turns[-1].id,
        turns=list(turns),
        total_tokens=tokens,
        entities=merge_entities(turns),
        summary=build_jev_summary(turns),
        is_lightweight=is_lightweight,
        is_foundational=is_foundational,
        has_tools=has_tools,
    )


def turn_has_tools(t: Turn) -> bool:
    for m in t.messages:
        if is_tool_result_message(m):
            return True
        if m.role == "assistant" and "tool_use" in m.content_string():
            return True
        if m.tool_calls:
            return True
    return False


def build_session_map(turns: List[Turn]) -> str:
    lines = ["Session Macro Timeline:"]
    for t in turns:
        user_prompt = ""
        for m in t.messages:
            if m.role == "user" and not is_tool_result_message(m):
                clean = extract_real_user_query(m.content_string())
                if len(clean) > 80:
                    clean = clean[:80] + "..."
                user_prompt = clean
                break
        if not user_prompt:
            user_prompt = "tool execution"
        lines.append(f'- Turn {t.id}: "{user_prompt}"')
    return "\n".join(lines)


def build_jev_summary(turns: List[Turn]) -> List[Dict[str, Any]]:
    out = []
    for turn in turns:
        user_prompt = ""
        assistant_summary = ""
        tool_output_snippet = ""
        tool_actions: List[str] = []

        for m in turn.messages:
            if m.role == "user" and not is_tool_result_message(m) and not user_prompt:
                clean = extract_real_user_query(m.content_string())
                if len(clean) > 200:
                    clean = clean[:200] + "..."
                user_prompt = clean
            elif m.role == "assistant" and not assistant_summary:
                clean = m.content_string()
                if len(clean) > 200:
                    clean = clean[:200] + "..."
                assistant_summary = clean

            if is_tool_result_message(m) and not tool_output_snippet:
                clean = m.content_string().strip()
                if len(clean) > 250:
                    clean = clean[:250] + "..."
                tool_output_snippet = clean

            if m.tool_calls:
                for tc in m.tool_calls:
                    if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
                        fn = tc["function"].get("name", "")
                        if fn:
                            tool_actions.append(fn)

        summary_parts = []
        if user_prompt:
            summary_parts.append(f"User: {user_prompt}")
        if tool_actions:
            deduped = sorted(list(set(tool_actions)))
            summary_parts.append(f"Tools: [{', '.join(deduped)}]")
        if assistant_summary:
            summary_parts.append(f"Assistant: {assistant_summary}")
        if tool_output_snippet:
            summary_parts.append(f"Tool Output: {tool_output_snippet}")
        if not summary_parts:
            summary_parts.append(f"Turn {turn.id} ({turn.token_count} tokens of context)")

        summary_text = " | ".join(summary_parts)
        out.append(
            {
                "turn_id": turn.id,
                "role": turn.role,
                "tokens": turn.token_count,
                "summary": sanitize_secrets(summary_text),
            }
        )
    return out


def merge_entities(turns: List[Turn]) -> EntityFootprint:
    files = set()
    symbols = set()
    errors = set()
    for t in turns:
        files.update(t.entities.files)
        symbols.update(t.entities.symbols)
        errors.update(t.entities.errors)
    return EntityFootprint(
        files=sorted(list(files)),
        symbols=sorted(list(symbols)),
        errors=sorted(list(errors)),
    )
