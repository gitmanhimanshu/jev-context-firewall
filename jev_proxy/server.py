import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from jev_proxy.causal_graph import CausalResolver
from jev_proxy.chunker import CandidateChunk, Chunker
from jev_proxy.config import Config
from jev_proxy.dashboard import DASHBOARD_HTML, global_tracker
from jev_proxy.jev_client import (
    ACTION_KEEP_FULL,
    EvaluationResult,
    JevClient,
)
from jev_proxy.parser import (
    Message,
    estimate_tokens,
    extract_real_user_query,
    group_messages_into_turns,
    is_tool_result_message,
    stitch_messages,
)
from jev_proxy.router import ProviderRouter, RouteInfo
from jev_proxy.tail_optimizer import TailOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jev_proxy")


class ProxyServer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.jev_client = JevClient(cfg)
        self.router = ProviderRouter(cfg)
        self.chunker = Chunker(cfg)
        self.causal_resolver = CausalResolver()
        self.tail_optimizer = TailOptimizer(cfg)
        self.semaphore = asyncio.Semaphore(cfg.get_concurrency())
        self.upstream_client = httpx.AsyncClient(
            timeout=None,  # No timeout for upstream streaming
            limits=httpx.Limits(max_keepalive_connections=128, max_connections=256),
        )

        self.app = FastAPI(title="Jev Context Firewall", docs_url=None, redoc_url=None)
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}

        @self.app.get("/dashboard", response_class=HTMLResponse)
        async def dashboard():
            return HTMLResponse(content=DASHBOARD_HTML)

        @self.app.get("/api/stats")
        async def stats():
            return global_tracker.get_metrics_dict()

        @self.app.get("/api/stats/export")
        async def export_stats():
            metrics = global_tracker.get_metrics_dict()
            report = {
                "report_timestamp": datetime.now().isoformat(),
                "proxy_engine": "Jev Context Firewall (Python)",
                "metrics": metrics,
            }
            return JSONResponse(
                content=report,
                headers={"Content-Disposition": 'attachment; filename="jev_firewall_report.json"'},
            )

        @self.app.get("/api/dump/latest")
        async def dump_latest():
            path = os.path.join("dumps", "latest_incoming.json")
            if not os.path.exists(path):
                return JSONResponse(status_code=404, content={"error": "No dumps recorded yet"})
            with open(path, "r", encoding="utf-8") as f:
                return JSONResponse(content=json.load(f))

        @self.app.get("/api/dump/pruned")
        async def dump_pruned():
            path = os.path.join("dumps", "latest_pruned.json")
            if not os.path.exists(path):
                return JSONResponse(status_code=404, content={"error": "No pruned dumps recorded yet"})
            with open(path, "r", encoding="utf-8") as f:
                return JSONResponse(content=json.load(f))

        @self.app.post("/{full_path:path}")
        async def proxy_catch_all(full_path: str, request: Request):
            return await self.handle_proxy(request, "/" + full_path)

    async def handle_proxy(self, request: Request, path: str) -> Response:
        start_time = time.time()
        body_bytes = await request.body()

        route = self.router.resolve_route(
            headers=dict(request.headers),
            query_params=dict(request.query_params),
            req_path=path,
        )

        pruned_bytes, orig_tok, pruned_tok, status = await self.process_and_prune_payload(
            path, body_bytes
        )

        self.save_payload_dump(body_bytes, pruned_bytes)

        latency_ms = int((time.time() - start_time) * 1000)
        return await self.forward_upstream(
            request=request,
            route=route,
            body=pruned_bytes,
            orig_tokens=orig_tok,
            pruned_tokens=pruned_tok,
            latency_ms=latency_ms,
            status=status,
            path=path,
        )

    def save_payload_dump(self, raw: bytes, pruned: bytes) -> None:
        os.makedirs("dumps", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            parsed = json.loads(raw)
            with open("dumps/latest_incoming.json", "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)
            with open(f"dumps/incoming_{ts}.json", "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)
        except Exception:
            with open("dumps/latest_incoming.json", "wb") as f:
                f.write(raw)

        if pruned:
            try:
                parsed_pruned = json.loads(pruned)
                with open("dumps/latest_pruned.json", "w", encoding="utf-8") as f:
                    json.dump(parsed_pruned, f, indent=2)
                with open(f"dumps/pruned_{ts}.json", "w", encoding="utf-8") as f:
                    json.dump(parsed_pruned, f, indent=2)
            except Exception:
                with open("dumps/latest_pruned.json", "wb") as f:
                    f.write(pruned)

    async def process_and_prune_payload(
        self, path: str, raw_body: bytes
    ) -> Tuple[bytes, int, int, str]:
        text_body = raw_body.decode("utf-8", errors="replace")
        orig_tokens = estimate_tokens(text_body)

        try:
            payload: Dict[str, Any] = json.loads(text_body)
        except Exception:
            return raw_body, orig_tokens, orig_tokens, "passthrough_raw"

        is_gemini_native = False
        raw_messages: List[Any] = []

        if isinstance(payload.get("messages"), list):
            raw_messages = payload["messages"]
        elif isinstance(payload.get("contents"), list):
            is_gemini_native = True
            if isinstance(payload.get("systemInstruction"), dict):
                parts = payload["systemInstruction"].get("parts")
                if isinstance(parts, list):
                    raw_messages.append({"role": "system", "content": parts})
            for item in payload["contents"]:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    if role == "model":
                        role = "assistant"
                    raw_messages.append({"role": role, "content": item.get("parts", [])})

        if len(raw_messages) < 2:
            return raw_body, orig_tokens, orig_tokens, "fast_path_bypass"

        messages: List[Message] = []
        for rm in raw_messages:
            if isinstance(rm, dict):
                messages.append(
                    Message(
                        role=rm.get("role", "user"),
                        content=rm.get("content", ""),
                        tool_calls=rm.get("tool_calls"),
                        tool_call_id=rm.get("tool_call_id"),
                        name=rm.get("name"),
                    )
                )

        current_query = ""
        for m in reversed(messages):
            if m.role == "user" and not is_tool_result_message(m):
                current_query = extract_real_user_query(m.content_string())
                break

        system_messages, turns = group_messages_into_turns(messages)

        tail_turns, candidate_chunks, _ = self.chunker.partition(turns, current_query)
        if not candidate_chunks:
            return raw_body, orig_tokens, orig_tokens, "fast_path_bypass"

        optimized_tail = self.tail_optimizer.optimize_tail(tail_turns, current_query)

        # Parallel OpenJev 6D Evaluation using Concurrency Semaphore
        eval_results: List[Optional[EvaluationResult]] = [None] * len(candidate_chunks)

        async def eval_single_chunk(idx: int, chunk: CandidateChunk) -> None:
            if chunk.is_lightweight:
                eval_results[idx] = EvaluationResult(
                    chunk_id=chunk.id,
                    selected=True,
                    action=ACTION_KEEP_FULL,
                    reason="lightweight_fast_path_approved",
                )
                return

            async with self.semaphore:
                state = {
                    "current_query": current_query,
                    "chunk_id": chunk.id,
                    "step_range": f"Turns {chunk.start_turn_id} to {chunk.end_turn_id}",
                    "chunk_summary": chunk.summary,
                    "entities": chunk.entities.to_dict(),
                    "has_tools": chunk.has_tools,
                    "is_foundational": chunk.is_foundational and not chunk.has_tools,
                }
                res = await self.jev_client.evaluate_batch(chunk.id, state)
                eval_results[idx] = res

        tasks = [eval_single_chunk(i, c) for i, c in enumerate(candidate_chunks)]
        await asyncio.gather(*tasks)

        final_evals = [e for e in eval_results if e is not None]
        resolved_chunks = self.causal_resolver.resolve_decisions(candidate_chunks, final_evals)

        final_content_messages: List[Message] = []
        for res in resolved_chunks:
            if res.action == ACTION_KEEP_FULL:
                for turn in res.chunk.turns:
                    final_content_messages.extend(turn.messages)

        for turn in optimized_tail:
            final_content_messages.extend(turn.messages)

        stitched = stitch_messages(system_messages, final_content_messages)

        if is_gemini_native:
            new_contents: List[Dict[str, Any]] = []
            for m in stitched:
                if m.role == "system":
                    continue
                g_role = "model" if m.role in ("assistant", "model") else "user"
                parts = m.content if isinstance(m.content, list) else [{"text": m.content_string()}]
                new_contents.append({"role": g_role, "parts": parts})
            payload["contents"] = new_contents
        else:
            payload["messages"] = [m.to_dict() for m in stitched]

        pruned_bytes = json.dumps(payload).encode("utf-8")
        pruned_tokens = estimate_tokens(pruned_bytes.decode("utf-8", errors="replace"))

        return pruned_bytes, orig_tokens, pruned_tokens, "pruned_success"

    async def forward_upstream(
        self,
        request: Request,
        route: RouteInfo,
        body: bytes,
        orig_tokens: int,
        pruned_tokens: int,
        latency_ms: int,
        status: str,
        path: str,
    ) -> Response:
        # Build upstream headers
        fwd_headers = {}
        excluded = {
            "host",
            "content-length",
            "x-upstream-url",
            "x-base-url",
            "x-target-url",
            "x-provider",
        }

        for k, v in request.headers.items():
            if k.lower() not in excluded:
                fwd_headers[k] = v

        for k, v in self.cfg.upstream_headers.items():
            if k not in fwd_headers and v:
                fwd_headers[k] = v

        # Synchronize Anthropic authentication headers
        auth_header = fwd_headers.get("authorization", "")
        api_key_header = fwd_headers.get("x-api-key", "")
        if not api_key_header and auth_header.startswith("Bearer "):
            fwd_headers["x-api-key"] = auth_header[len("Bearer "):]
        elif not auth_header and api_key_header:
            fwd_headers["authorization"] = f"Bearer {api_key_header}"

        if (
            route.provider == "anthropic" or "messages" in route.target_url.lower()
        ) and "anthropic-version" not in fwd_headers:
            fwd_headers["anthropic-version"] = "2023-06-01"

        fwd_headers["content-length"] = str(len(body))

        try:
            req = self.upstream_client.build_request(
                method="POST",
                url=route.target_url,
                headers=fwd_headers,
                content=body,
            )
            upstream_resp = await self.upstream_client.send(req, stream=True)
        except Exception as e:
            logger.error("Upstream connection error: %v", e)
            return JSONResponse(
                status_code=502,
                content={"error": f"Upstream connection error: {e}"},
            )

        content_type = upstream_resp.headers.get("content-type", "").lower()
        is_stream = "text/event-stream" in content_type

        resp_headers = {}
        for k, v in upstream_resp.headers.items():
            if is_stream and k.lower() == "content-length":
                continue
            resp_headers[k] = v

        if is_stream:
            resp_headers["cache-control"] = "no-cache, no-transform"
            resp_headers["connection"] = "keep-alive"
            resp_headers["x-accel-buffering"] = "no"

        async def stream_generator():
            try:
                async for chunk in upstream_resp.aiter_bytes():
                    yield chunk
            finally:
                await upstream_resp.aclose()

        await global_tracker.record_request(
            path=path,
            original_tokens=orig_tokens,
            pruned_tokens=pruned_tokens,
            latency_ms=latency_ms,
            status=status,
        )

        red_pct = 0.0
        if orig_tokens > 0:
            red_pct = float(orig_tokens - pruned_tokens) / float(orig_tokens) * 100.0

        client_ip = request.client.host if request.client else "127.0.0.1"
        logger.info(
            "[PROXY] [%s] %s -> %s | %d -> %d tokens (%.1f%% saved) | Latency: %dms | Status: %s | HTTP %d",
            client_ip,
            route.provider,
            route.target_url,
            orig_tokens,
            pruned_tokens,
            red_pct,
            latency_ms,
            status,
            upstream_resp.status_code,
        )

        return StreamingResponse(
            stream_generator(),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
        )
