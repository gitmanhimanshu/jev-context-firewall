import logging
from dataclasses import dataclass
from typing import List

from jev_proxy.chunker import CandidateChunk
from jev_proxy.jev_client import ACTION_EVICT, ACTION_KEEP_FULL, EvaluationResult

logger = logging.getLogger("jev_proxy.causal_graph")


@dataclass
class ResolvedChunk:
    chunk: CandidateChunk
    action: str  # ACTION_KEEP_FULL, ACTION_COMPRESS
    reason: str


class CausalResolver:
    def resolve_decisions(
        self,
        chunks: List[CandidateChunk],
        evaluations: List[EvaluationResult],
    ) -> List[ResolvedChunk]:
        if not chunks:
            return []

        eval_map = {ev.chunk_id: ev for ev in evaluations}
        resolved: List[ResolvedChunk] = []

        for chunk in chunks:
            ev = eval_map.get(chunk.id)
            if not ev:
                resolved.append(
                    ResolvedChunk(
                        chunk=chunk,
                        action=ACTION_KEEP_FULL,
                        reason="missing_eval_fail_open",
                    )
                )
                continue

            if ev.action == ACTION_EVICT or (not ev.selected and not ev.action):
                logger.info(
                    "[Jev Cognitive Firewall] Evicted Chunk %d (noul: %.3f, topic: %s, dep: %s, life: %s, waste: %s, reason: %s)",
                    chunk.id,
                    ev.noul,
                    ev.topic,
                    ev.context_dep,
                    ev.lifecycle,
                    ev.token_waste,
                    ev.reason,
                )
                continue

            logger.info(
                "[Jev Cognitive Firewall] Approved Chunk %d -> %s (noul: %.3f, topic: %s, reason: %s)",
                chunk.id,
                ev.action,
                ev.noul,
                ev.topic,
                ev.reason,
            )

            resolved.append(
                ResolvedChunk(
                    chunk=chunk,
                    action=ev.action,
                    reason=ev.reason,
                )
            )

        return resolved

    def resolve_dependencies(
        self,
        chunks: List[CandidateChunk],
        evaluations: List[EvaluationResult],
    ) -> List[CandidateChunk]:
        decisions = self.resolve_decisions(chunks, evaluations)
        return [d.chunk for d in decisions]
