import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx

from jev_proxy.config import Config

ACTION_KEEP_FULL = "KEEP_FULL"
ACTION_COMPRESS = "COMPRESS"
ACTION_EVICT = "EVICT"


@dataclass
class EvaluationResult:
    chunk_id: int
    selected: bool
    action: str
    reason: str
    noul: float = 0.0
    topic: str = ""
    context_dep: str = ""
    lifecycle: str = ""
    token_waste: str = ""
    preservation_target: str = ""
    latency_ms: int = 0
    error: Optional[str] = None


class JevClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._key_index = 0
        self._lock = asyncio.Lock()
        self.http_client = httpx.AsyncClient(
            timeout=float(cfg.jev_timeout_ms) / 1000.0,
            limits=httpx.Limits(max_keepalive_connections=128, max_connections=256),
        )

    async def next_api_key(self) -> str:
        keys = self.cfg.get_api_keys()
        if not keys:
            return self.cfg.jev_api_key
        async with self._lock:
            idx = self._key_index
            self._key_index += 1
            return keys[idx % len(keys)]

    async def evaluate_batch(self, chunk_id: int, state: Any) -> EvaluationResult:
        start_time = time.time()
        url = f"{self.cfg.jev_base_url}/v1/systemone"

        req_body = {
            "model": self.cfg.jev_model,
            "state": state,
            "questions": {
                "relevance_score": {
                    "type": "noul",
                    "instructions": "Is this historical content directly relevant or needed to fulfill the active user query?",
                    "criteria": {
                        "true": "Directly relevant, required data, or project rule needed for the active query.",
                        "false": "Unrelated past tool output, obsolete file content, or past discussion not needed for this query.",
                    },
                },
                "topic_relationship": {
                    "type": "choice",
                    "instructions": "How does this content relate to the active user query?",
                    "criteria": {
                        "identical_thread": "Directly part of fulfilling this exact query.",
                        "shared_background": "Shares related project context or configuration.",
                        "completely_disjoint": "Completely unrelated task, different file, or finished past query.",
                    },
                },
                "context_dependency": {
                    "type": "choice",
                    "instructions": "Would omitting this historical content cause errors, contradict instructions, or lose critical data needed for the active query?",
                    "criteria": {
                        "critical_loss": "Removing it breaks continuity, contradicts instructions, or loses critical facts or code needed for the active query.",
                        "minor_context": "Removing it loses minor historical nuance, but the core task can be completed accurately without it.",
                        "zero_loss": "Zero impact; the active query can be fulfilled completely and accurately without this content.",
                    },
                },
                "lifecycle_state": {
                    "type": "choice",
                    "instructions": "What is the lifecycle state of this historical step in the overall session?",
                    "criteria": {
                        "foundational_rule": "A global rule, user preference, system directive, or setup constraint that applies across all turns.",
                        "active_thread": "Part of the active subtask or working thread in progress.",
                        "completed_subtask": "A past subtask, query, or tool output that has concluded or was superseded.",
                        "irrelevant_tangent": "An unrelated tangent, dead-end exchange, or obsolete conversation.",
                    },
                },
                "token_cost_waste": {
                    "type": "choice",
                    "instructions": "Considering the volume of text, data, or tool outputs in this step, how redundant or verbose is it relative to its contextual value for the active query?",
                    "criteria": {
                        "heavy_waste": "Contains voluminous outputs, repetitive data, or large blocks with little to no utility for the active query.",
                        "moderate_cost": "Moderate length with some contextual value.",
                        "essential_tokens": "Compact, high-signal information that should be retained intact.",
                    },
                },
                "preservation_target": {
                    "type": "choice",
                    "instructions": "Considering context efficiency and answer quality, how should this ~1000-token chunk be handled?",
                    "criteria": {
                        "keep_full_detail": "Retain this chunk in the context window.",
                        "drop_completely": "Safely omit this chunk to eliminate token waste.",
                    },
                },
            },
        }

        api_key = await self.next_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        resp = None
        latency_ms = 0
        try:
            resp = await self.http_client.post(url, json=req_body, headers=headers)
            latency_ms = int((time.time() - start_time) * 1000)

            # Retry once on 429 if multi-key pool exists
            if resp.status_code == 429 and len(self.cfg.get_api_keys()) > 1:
                retry_key = await self.next_api_key()
                if retry_key == api_key and len(self.cfg.get_api_keys()) > 1:
                    retry_key = await self.next_api_key()
                headers["Authorization"] = f"Bearer {retry_key}"
                resp = await self.http_client.post(url, json=req_body, headers=headers)
                latency_ms = int((time.time() - start_time) * 1000)

        except Exception as e:
            # Fail-open on timeout or connection error
            latency_ms = int((time.time() - start_time) * 1000)
            return EvaluationResult(
                chunk_id=chunk_id,
                selected=True,
                action=ACTION_KEEP_FULL,
                reason=f"fail_open_timeout_or_network ({e})",
                latency_ms=latency_ms,
                error=str(e),
            )

        if resp is None or resp.status_code != 200:
            status_code = resp.status_code if resp is not None else 0
            return EvaluationResult(
                chunk_id=chunk_id,
                selected=True,
                action=ACTION_KEEP_FULL,
                reason=f"fail_open_http_{status_code}",
                latency_ms=latency_ms,
                error=f"status {status_code}",
            )

        try:
            data = resp.json()
            answers = data.get("answers", {})

            noul_score = 0.0
            if "relevance_score" in answers and "noul" in answers["relevance_score"]:
                noul_score = float(answers["relevance_score"]["noul"] or 0.0)

            def get_choice(k: str, fallback: str) -> str:
                if k in answers and "choice" in answers[k] and answers[k]["choice"]:
                    return str(answers[k]["choice"])
                return fallback

            topic = get_choice("topic_relationship", "completely_disjoint")
            context_dep = get_choice("context_dependency", "zero_loss")
            lifecycle = get_choice("lifecycle_state", "completed_subtask")
            token_waste = get_choice("token_cost_waste", "heavy_waste")
            preservation = get_choice("preservation_target", "drop_completely")

            action, reason = arbitrate_consensus(
                noul_score,
                topic,
                context_dep,
                lifecycle,
                token_waste,
                preservation,
                self.cfg.relevance_threshold,
            )

            # Low confidence boundary protection
            if (
                action == ACTION_EVICT
                and noul_score >= self.cfg.relevance_threshold
                and topic != "completely_disjoint"
                and preservation != "drop_completely"
            ):
                crit_conf = 1.0
                if "topic_relationship" in answers and answers["topic_relationship"].get("confidence") is not None:
                    crit_conf = float(answers["topic_relationship"]["confidence"])
                if (
                    "preservation_target" in answers
                    and answers["preservation_target"].get("confidence") is not None
                    and float(answers["preservation_target"]["confidence"]) < crit_conf
                ):
                    crit_conf = float(answers["preservation_target"]["confidence"])

                if crit_conf < 0.35:
                    action = ACTION_KEEP_FULL
                    reason = f"keep_full_low_confidence_boundary_{crit_conf:.2f}"

            return EvaluationResult(
                chunk_id=chunk_id,
                selected=(action != ACTION_EVICT),
                action=action,
                reason=reason,
                noul=noul_score,
                topic=topic,
                context_dep=context_dep,
                lifecycle=lifecycle,
                token_waste=token_waste,
                preservation_target=preservation,
                latency_ms=latency_ms,
            )
        except Exception as e:
            return EvaluationResult(
                chunk_id=chunk_id,
                selected=True,
                action=ACTION_KEEP_FULL,
                reason="fail_open_decode_error",
                latency_ms=latency_ms,
                error=str(e),
            )


def arbitrate_consensus(
    noul: float,
    topic: str,
    context_dep: str,
    lifecycle: str,
    token_waste: str,
    preservation: str,
    threshold: float,
) -> Tuple[str, str]:
    # Guard 1: Foundational Rule Protection
    if lifecycle == "foundational_rule" and noul >= 0.50:
        return ACTION_KEEP_FULL, "foundational_rule_full"

    # Rule 2: Explicit Drop Target
    if preservation == "drop_completely":
        return ACTION_EVICT, f"evicted_drop_target_noul_{noul:.3f}"

    # Rule 3: Completely Disjoint or Tangent
    if topic == "completely_disjoint" and noul < threshold:
        return ACTION_EVICT, f"evicted_disjoint_noul_{noul:.3f}"
    if lifecycle == "irrelevant_tangent" and noul < threshold:
        return ACTION_EVICT, f"evicted_tangent_noul_{noul:.3f}"

    # Rule 4: Zero Loss or Heavy Waste
    if context_dep == "zero_loss" and noul < threshold:
        return ACTION_EVICT, f"evicted_zero_loss_noul_{noul:.3f}"
    if token_waste == "heavy_waste" and noul < threshold:
        return ACTION_EVICT, f"evicted_heavy_waste_noul_{noul:.3f}"

    # Rule 5: Keep High Utility
    if noul >= threshold and (context_dep == "critical_loss" or topic == "identical_thread"):
        return ACTION_KEEP_FULL, f"keep_full_critical_noul_{noul:.3f}"

    # Rule 6: Below Threshold Eviction
    if noul < threshold:
        return ACTION_EVICT, f"evicted_below_threshold_noul_{noul:.3f}"

    return ACTION_KEEP_FULL, f"keep_full_default_noul_{noul:.3f}"
