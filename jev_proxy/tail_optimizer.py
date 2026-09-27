import json
from typing import Dict, List, Optional, Set

from jev_proxy.config import Config
from jev_proxy.parser import Message, Turn, estimate_tokens, is_tool_result_message


class TailOptimizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def optimize_tail(self, tail_turns: List[Turn], current_query: str) -> List[Turn]:
        return self.optimize_turn_sequence(tail_turns, self.cfg.sacrosanct_tail_size)

    def optimize_turn_sequence(self, turns: List[Turn], sacrosanct_count: int) -> List[Turn]:
        if not turns:
            return []

        split_idx = len(turns) - sacrosanct_count
        if split_idx < 0:
            split_idx = 0

        optimizable_turns = turns[:split_idx]
        sacrosanct_turns = turns[split_idx:]

        optimized: List[Turn] = []
        for i, turn in enumerate(optimizable_turns):
            later_edited_files: Set[str] = set()
            for j in range(i + 1, len(turns)):
                for f in turns[j].entities.files:
                    later_edited_files.add(f.lower())
            optimized.append(self._optimize_turn(turn, later_edited_files, False))

        for i, turn in enumerate(sacrosanct_turns):
            is_very_last_turn = (i == len(sacrosanct_turns) - 1)
            optimized.append(self._optimize_turn(turn, None, is_very_last_turn))

        return optimized

    def _optimize_turn(
        self,
        turn: Turn,
        later_edited_files: Optional[Set[str]],
        preserve_last_message_only: bool,
    ) -> Turn:
        clean_messages: List[Message] = []
        is_superseded_turn = False
        superseded_file = ""

        if later_edited_files:
            for f in turn.entities.files:
                if f.lower() in later_edited_files:
                    is_superseded_turn = True
                    superseded_file = f
                    break

        last_idx = len(turn.messages) - 1
        for idx, msg in enumerate(turn.messages):
            if preserve_last_message_only and idx == last_idx:
                clean_messages.append(msg)
                continue

            if is_tool_result_message(msg):
                clean_messages.append(
                    self._optimize_tool_message(msg, is_superseded_turn, superseded_file)
                )
            elif msg.role == "assistant" and not preserve_last_message_only:
                clean_messages.append(self._optimize_assistant_message(msg))
            else:
                clean_messages.append(msg)

        tokens = sum(estimate_tokens(m.content_string()) for m in clean_messages)
        return Turn(
            id=turn.id,
            role=turn.role,
            messages=clean_messages,
            entities=turn.entities,
            token_count=tokens,
        )

    def _optimize_tool_message(
        self,
        msg: Message,
        is_superseded: bool,
        superseded_file: str,
    ) -> Message:
        if isinstance(msg.content, list):
            new_blocks: List[dict] = []
            for item in msg.content:
                if not isinstance(item, dict):
                    new_blocks.append(item)
                    continue

                # Gemini functionResponse
                if "functionResponse" in item and isinstance(item["functionResponse"], dict):
                    raw_resp = json.dumps(item["functionResponse"].get("response", {}))
                    tokens = estimate_tokens(raw_resp)
                    if is_superseded and tokens > 400:
                        new_item = dict(item)
                        new_item["functionResponse"] = {
                            "name": item["functionResponse"].get("name", ""),
                            "response": {
                                "result": f"[Tool Result: read of {superseded_file} ({tokens} tokens) superseded by later edits]"
                            },
                        }
                        new_blocks.append(new_item)
                        continue
                    new_blocks.append(item)
                    continue

                # Anthropic tool_result
                if item.get("type") == "tool_result":
                    c_val = item.get("content", "")
                    c_str = c_val if isinstance(c_val, str) else json.dumps(c_val)
                    tokens = estimate_tokens(c_str)
                    if is_superseded and tokens > 400:
                        new_item = dict(item)
                        new_item["content"] = f"[Tool Result: read of {superseded_file} ({tokens} tokens) superseded by later edits]"
                        new_blocks.append(new_item)
                        continue
                    new_blocks.append(item)
                    continue

                new_blocks.append(item)

            return Message(
                role=msg.role,
                content=new_blocks,
                tool_calls=msg.tool_calls,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            )

        # Plain text content (OpenAI format)
        content_str = msg.content_string()
        tokens = estimate_tokens(content_str)
        if is_superseded and tokens > 400:
            return Message(
                role=msg.role,
                content=f"[Tool Result: read of {superseded_file} ({tokens} tokens) superseded by later edits]",
                tool_calls=msg.tool_calls,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            )

        return msg

    def _optimize_assistant_message(self, msg: Message) -> Message:
        if isinstance(msg.content, list):
            new_blocks: List[dict] = []
            for item in msg.content:
                if not isinstance(item, dict):
                    new_blocks.append(item)
                    continue

                # NEVER compress tool_use (Anthropic) or functionCall (Gemini) blocks
                if item.get("type") == "tool_use" or "functionCall" in item:
                    new_blocks.append(item)
                    continue

                if isinstance(item.get("text"), str) and len(item["text"]) > 600:
                    new_item = dict(item)
                    new_item["text"] = item["text"][:350] + "\n\n[... Previous assistant response compressed by Jev Proxy ...]"
                    new_blocks.append(new_item)
                    continue

                new_blocks.append(item)

            return Message(
                role=msg.role,
                content=new_blocks,
                tool_calls=msg.tool_calls,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            )

        content_str = msg.content_string()
        if len(content_str) > 600:
            return Message(
                role=msg.role,
                content=content_str[:350] + "\n\n[... Previous assistant response compressed by Jev Proxy ...]",
                tool_calls=msg.tool_calls,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            )

        return msg
