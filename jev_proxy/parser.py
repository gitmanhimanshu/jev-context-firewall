import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from jev_proxy.extractor import EntityFootprint, extract_entities


@dataclass
class ToolCall:
    id: str = ""
    type: str = "function"
    function: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"id": self.id, "type": self.type, "function": self.function}
        return d


@dataclass
class Message:
    role: str
    content: Any  # str or list
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def content_string(self) -> str:
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            sb = []
            for item in self.content:
                if isinstance(item, dict):
                    if "text" in item and isinstance(item["text"], str):
                        sb.append(item["text"])
                    if "content" in item:
                        if isinstance(item["content"], str):
                            sb.append(item["content"])
                        elif isinstance(item["content"], list):
                            for cb in item["content"]:
                                if isinstance(cb, dict) and "text" in cb and isinstance(cb["text"], str):
                                    sb.append(cb["text"])
                    if "input" in item:
                        sb.append(json.dumps(item["input"]))
                    if "functionCall" in item:
                        sb.append(json.dumps(item["functionCall"]))
                    if "functionResponse" in item:
                        sb.append(json.dumps(item["functionResponse"]))
                else:
                    sb.append(str(item))
            return "\n".join(sb)
        elif self.content is not None:
            return json.dumps(self.content)
        return ""

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class Turn:
    id: int
    role: str
    messages: List[Message]
    entities: EntityFootprint
    token_count: int = 0


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Anthropic Claude BPE tokenizer encodes structured JSON, tool schemas, and code at ~2.36 bytes per token
    tokens = int(len(text) / 2.36)
    return tokens if tokens > 0 else 1


def extract_real_user_query(content: str) -> str:
    start_tag = "<user_input>"
    end_tag = "</user_input>"
    if start_tag in content:
        start_idx = content.find(start_tag)
        end_idx = content.find(end_tag, start_idx)
        if end_idx != -1:
            clean = content[start_idx + len(start_tag): end_idx].strip()
            if clean:
                return clean
    return (content or "").strip()


def is_tool_result_message(m: Message) -> bool:
    if m.role in ("tool", "function") or m.tool_call_id:
        return True
    if isinstance(m.content, list):
        for b in m.content:
            if isinstance(b, dict):
                if b.get("type") == "tool_result" or "functionResponse" in b:
                    return True
    return False


def has_tool_use(m: Message) -> bool:
    if m.tool_calls and len(m.tool_calls) > 0:
        return True
    if isinstance(m.content, list):
        for b in m.content:
            if isinstance(b, dict):
                if b.get("type") == "tool_use" or "functionCall" in b:
                    return True
    return False


def get_tool_use_ids(m: Message) -> Set[str]:
    ids = set()
    if m.tool_calls:
        for tc in m.tool_calls:
            if isinstance(tc, dict) and tc.get("id"):
                ids.add(tc["id"])
    if isinstance(m.content, list):
        for b in m.content:
            if isinstance(b, dict):
                if b.get("type") == "tool_use" and b.get("id"):
                    ids.add(b["id"])
                if "functionCall" in b and isinstance(b["functionCall"], dict):
                    fc = b["functionCall"]
                    if fc.get("id"):
                        ids.add(fc["id"])
                    if fc.get("name"):
                        ids.add(fc["name"])
    return ids


def get_tool_result_ids(m: Message) -> Set[str]:
    ids = set()
    if m.role in ("tool", "function") and m.tool_call_id:
        ids.add(m.tool_call_id)
    if isinstance(m.content, list):
        for b in m.content:
            if isinstance(b, dict):
                if b.get("type") == "tool_result" and b.get("tool_use_id"):
                    ids.add(b["tool_use_id"])
                if "functionResponse" in b and isinstance(b["functionResponse"], dict):
                    fr = b["functionResponse"]
                    if fr.get("id"):
                        ids.add(fr["id"])
                    if fr.get("name"):
                        ids.add(fr["name"])
    return ids


def group_messages_into_turns(messages: List[Message]) -> Tuple[List[Message], List[Turn]]:
    system_messages: List[Message] = []
    non_system: List[Message] = []

    for m in messages:
        if m.role in ("system", "developer"):
            system_messages.append(m)
        else:
            non_system.append(m)

    turns: List[Turn] = []
    current_turn: Optional[Turn] = None

    for msg in non_system:
        is_real_user_prompt = (msg.role == "user" and not is_tool_result_message(msg))
        is_user_turn_start = is_real_user_prompt and (
            current_turn is None or current_turn.role != "user" or len(current_turn.messages) > 1
        )

        if is_user_turn_start and current_turn is not None and len(current_turn.messages) > 0:
            finalize_turn(current_turn)
            turns.append(current_turn)
            current_turn = None

        if current_turn is None:
            current_turn = Turn(
                id=len(turns),
                role=msg.role,
                messages=[],
                entities=EntityFootprint(),
            )

        current_turn.messages.append(msg)

    if current_turn is not None and len(current_turn.messages) > 0:
        finalize_turn(current_turn)
        turns.append(current_turn)

    return system_messages, turns


def finalize_turn(t: Turn) -> None:
    sb = []
    tokens = 0

    for m in t.messages:
        content_str = m.content_string()
        sb.append(content_str)
        tokens += estimate_tokens(content_str)

        if m.tool_calls:
            raw_tc = json.dumps(m.tool_calls)
            sb.append(raw_tc)
            tokens += estimate_tokens(raw_tc)

    full_text = " ".join(sb)
    t.token_count = tokens
    t.entities = extract_entities(full_text)


def sanitize_unanswered_tool_calls(assistant_msg: Message, answered_ids: Set[str]) -> Message:
    # 1. OpenAI ToolCalls
    if assistant_msg.tool_calls:
        valid_calls = []
        dropped_notes = []
        for tc in assistant_msg.tool_calls:
            tc_id = tc.get("id", "")
            if tc_id and answered_ids and tc_id in answered_ids:
                valid_calls.append(tc)
            else:
                fn_name = ""
                if isinstance(tc.get("function"), dict):
                    fn_name = tc["function"].get("name", "")
                dropped_notes.append(f"[Historical Tool Call to {fn_name} (ID: {tc_id})]")
        assistant_msg.tool_calls = valid_calls if valid_calls else None
        if dropped_notes:
            notes_text = "\n".join(dropped_notes)
            s = assistant_msg.content_string()
            if not s:
                assistant_msg.content = notes_text
            else:
                assistant_msg.content = f"{s}\n\n{notes_text}"

    # 2. Anthropic & Gemini blocks in content list
    if isinstance(assistant_msg.content, list):
        clean_blocks = []
        modified = False
        for b in assistant_msg.content:
            if isinstance(b, dict):
                # Anthropic tool_use
                if b.get("type") == "tool_use":
                    tool_id = b.get("id", "")
                    name = b.get("name", "")
                    if not tool_id or not answered_ids or tool_id not in answered_ids:
                        raw_input = json.dumps(b.get("input", {}))
                        clean_blocks.append({
                            "type": "text",
                            "text": f"[Historical Tool Call to {name}: {raw_input}]"
                        })
                        modified = True
                        continue
                # Gemini functionCall
                if "functionCall" in b and isinstance(b["functionCall"], dict):
                    fc = b["functionCall"]
                    fc_id = fc.get("id", "")
                    name = fc.get("name", "")
                    is_answered = False
                    if fc_id and answered_ids and fc_id in answered_ids:
                        is_answered = True
                    elif name and answered_ids and name in answered_ids:
                        is_answered = True
                    if not is_answered:
                        raw_args = json.dumps(fc.get("args", {}))
                        clean_blocks.append({
                            "text": f"[Historical Function Call to {name}: {raw_args}]"
                        })
                        modified = True
                        continue
            clean_blocks.append(b)
        if modified:
            assistant_msg.content = clean_blocks

    return assistant_msg


def sanitize_orphaned_tool_results(msg: Message, valid_tool_ids: Optional[Set[str]]) -> Message:
    # OpenAI role: tool or function
    if msg.role in ("tool", "function"):
        if msg.tool_call_id and (valid_tool_ids is None or msg.tool_call_id not in valid_tool_ids):
            return Message(
                role="user",
                content=f"[Historical Tool Output for {msg.tool_call_id}: {msg.content_string()}]"
            )
        return msg

    # Anthropic & Gemini blocks in content list
    if isinstance(msg.content, list):
        clean_blocks = []
        modified = False
        for b in msg.content:
            if isinstance(b, dict):
                if b.get("type") == "tool_result":
                    tool_id = b.get("tool_use_id", "")
                    if tool_id and (valid_tool_ids is None or tool_id not in valid_tool_ids):
                        raw_content = json.dumps(b.get("content", ""))
                        clean_blocks.append({
                            "type": "text",
                            "text": f"[Historical Tool Output for {tool_id}: {raw_content}]"
                        })
                        modified = True
                        continue
                if "functionResponse" in b and isinstance(b["functionResponse"], dict):
                    fr = b["functionResponse"]
                    fr_id = fr.get("id", "")
                    name = fr.get("name", "")
                    is_valid = False
                    if fr_id and valid_tool_ids and fr_id in valid_tool_ids:
                        is_valid = True
                    elif name and valid_tool_ids and name in valid_tool_ids:
                        is_valid = True
                    if not is_valid:
                        raw_resp = json.dumps(fr.get("response", {}))
                        clean_blocks.append({
                            "text": f"[Historical Function Response for {name}: {raw_resp}]"
                        })
                        modified = True
                        continue
            clean_blocks.append(b)
        if modified:
            msg.content = clean_blocks

    return msg


def _is_gemini_part(bm: dict) -> bool:
    if "functionCall" in bm or "functionResponse" in bm:
        return True
    if "text" in bm and "type" not in bm:
        return True
    return False


def merge_consecutive_contents(m1: Message, m2: Message) -> Message:
    res = Message(
        role=m1.role,
        content=m1.content,
        tool_calls=list(m1.tool_calls) if m1.tool_calls else None,
        tool_call_id=m1.tool_call_id,
        name=m1.name,
    )

    if m2.tool_calls:
        if not res.tool_calls:
            res.tool_calls = []
        res.tool_calls.extend(m2.tool_calls)

    b1_is_list = isinstance(m1.content, list)
    b2_is_list = isinstance(m2.content, list)

    if b1_is_list or b2_is_list:
        is_gemini = False
        if b1_is_list:
            for item in m1.content:
                if isinstance(item, dict) and _is_gemini_part(item):
                    is_gemini = True
                    break
        if not is_gemini and b2_is_list:
            for item in m2.content:
                if isinstance(item, dict) and _is_gemini_part(item):
                    is_gemini = True
                    break

        blocks: List[Any] = []
        existing_tool_results: Dict[str, int] = {}
        existing_func_responses: Dict[str, int] = {}

        if b1_is_list:
            for b in m1.content:
                idx = len(blocks)
                blocks.append(b)
                if isinstance(b, dict):
                    if b.get("type") == "tool_result" and b.get("tool_use_id"):
                        existing_tool_results[b["tool_use_id"]] = idx
                    if "functionResponse" in b and isinstance(b["functionResponse"], dict):
                        name = b["functionResponse"].get("name", "")
                        if name:
                            existing_func_responses[name] = idx
        elif m1.content is not None:
            if is_gemini:
                blocks.append({"text": m1.content_string()})
            else:
                blocks.append({"type": "text", "text": m1.content_string()})

        if b2_is_list:
            for item in m2.content:
                if isinstance(item, dict):
                    if item.get("type") == "tool_result":
                        tid = item.get("tool_use_id", "")
                        if tid and tid in existing_tool_results:
                            exist_map = blocks[existing_tool_results[tid]]
                            exist_content = exist_map.get("content", "")
                            new_content = item.get("content", "")
                            exist_map["content"] = f"{exist_content}\n\n[... omitted by OpenJev ...]\n\n{new_content}"
                            continue
                        elif tid:
                            existing_tool_results[tid] = len(blocks)

                    if "functionResponse" in item and isinstance(item["functionResponse"], dict):
                        name = item["functionResponse"].get("name", "")
                        if name and name in existing_func_responses:
                            exist_map = blocks[existing_func_responses[name]]
                            if "functionResponse" in exist_map:
                                ex_fr = exist_map["functionResponse"]
                                ex_resp = json.dumps(ex_fr.get("response", {}))
                                new_resp = json.dumps(item["functionResponse"].get("response", {}))
                                ex_fr["response"] = {
                                    "result": f"{ex_resp}\n\n[... omitted by OpenJev ...]\n\n{new_resp}"
                                }
                            continue
                        elif name:
                            existing_func_responses[name] = len(blocks)
                blocks.append(item)
        elif m2.content is not None:
            if is_gemini:
                blocks.append({"text": m2.content_string()})
            else:
                blocks.append({"type": "text", "text": m2.content_string()})

        res.content = blocks
        return res

    s1 = m1.content_string()
    s2 = m2.content_string()
    if not s1:
        res.content = s2
    elif not s2:
        res.content = s1
    else:
        res.content = f"{s1}\n\n{s2}"
    return res


def stitch_messages(system_messages: List[Message], content_messages: List[Message]) -> List[Message]:
    out = list(system_messages or [])
    if not content_messages:
        return out

    # Step 1: Merge consecutive messages of the exact same role
    merged: List[Message] = []
    for m in content_messages:
        if not merged:
            merged.append(m)
            continue

        last = merged[-1]
        if last.role == m.role:
            if last.role in ("tool", "function") and last.tool_call_id and last.tool_call_id == m.tool_call_id:
                last.content = f"{last.content_string()}\n\n[... omitted by OpenJev ...]\n\n{m.content_string()}"
                continue
            if last.role not in ("tool", "function"):
                merged[-1] = merge_consecutive_contents(last, m)
                continue
        merged.append(m)

    # Step 2: Two-way Tool Pairing Validation across Anthropic, OpenAI, and Gemini
    validated: List[Message] = []
    n = len(merged)
    i = 0

    while i < n:
        m = merged[i]
        if m.role in ("assistant", "model"):
            answered_ids: Set[str] = set()
            for j in range(i + 1, n):
                next_msg = merged[j]
                if next_msg.role in ("assistant", "model"):
                    break
                if next_msg.role in ("tool", "function") and next_msg.tool_call_id:
                    answered_ids.add(next_msg.tool_call_id)
                elif next_msg.role == "user":
                    res_ids = get_tool_result_ids(next_msg)
                    answered_ids.update(res_ids)
                    break

            clean_assistant = sanitize_unanswered_tool_calls(m, answered_ids)
            validated.append(clean_assistant)

            active_tool_ids = get_tool_use_ids(clean_assistant)
            j = i + 1
            while j < n:
                sub_msg = merged[j]
                if sub_msg.role in ("assistant", "model"):
                    break
                clean_sub = sanitize_orphaned_tool_results(sub_msg, active_tool_ids)
                validated.append(clean_sub)
                j += 1
            i = j
        elif m.role in ("tool", "function"):
            clean_tool = sanitize_orphaned_tool_results(m, None)
            validated.append(clean_tool)
            i += 1
        else:
            clean_user = sanitize_orphaned_tool_results(m, None)
            validated.append(clean_user)
            i += 1

    # Step 3: Final role alternation pass
    final_messages: List[Message] = []
    for m in validated:
        if not final_messages:
            final_messages.append(m)
            continue

        last = final_messages[-1]
        if last.role == m.role:
            if last.role in ("tool", "function") and last.tool_call_id and last.tool_call_id == m.tool_call_id:
                last.content = f"{last.content_string()}\n\n[... omitted by OpenJev ...]\n\n{m.content_string()}"
                continue
            if last.role not in ("tool", "function"):
                final_messages[-1] = merge_consecutive_contents(last, m)
                continue
        final_messages.append(m)

    # Step 4: Ensure the first message starts with "user"
    if final_messages and final_messages[0].role in ("assistant", "model"):
        final_messages.insert(0, Message(role="user", content="[Continuing session context]"))

    return out + final_messages
