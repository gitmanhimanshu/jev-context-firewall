from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import parse_qs, urlencode, urlparse

from jev_proxy.config import Config


@dataclass
class RouteInfo:
    target_url: str
    provider: str
    is_custom: bool


class ProviderRouter:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def resolve_route(
        self,
        headers: Dict[str, str],
        query_params: Dict[str, str],
        req_path: str,
    ) -> RouteInfo:
        # Normalize header keys to lowercase
        norm_headers = {k.lower(): v for k, v in headers.items()}

        custom_upstream = (
            norm_headers.get("x-upstream-url", "")
            or norm_headers.get("x-base-url", "")
            or norm_headers.get("x-target-url", "")
        ).strip()

        # Check query parameter ?upstream= or ?target=
        forward_params = dict(query_params)
        if not custom_upstream:
            if "upstream" in forward_params:
                custom_upstream = forward_params.pop("upstream").strip()
            elif "target" in forward_params:
                custom_upstream = forward_params.pop("target").strip()
        else:
            forward_params.pop("upstream", None)
            forward_params.pop("target", None)

        # Provider override
        custom_provider = (
            norm_headers.get("x-provider", "")
            or norm_headers.get("x-api-format", "")
            or norm_headers.get("x-format", "")
            or forward_params.pop("provider", "")
            or forward_params.pop("format", "")
        ).strip().lower()

        forward_query = urlencode(forward_params)

        lower_path = req_path.lower()
        target_path = req_path
        provider = "openai"
        upstream_base = self._default_base("openai", "https://api.openai.com")

        # Path prefix matching
        if lower_path.startswith("/proxy/anthropic/"):
            provider = "anthropic"
            target_path = req_path[len("/proxy/anthropic"):]
            upstream_base = self._default_base("anthropic", "https://api.anthropic.com")
        elif lower_path.startswith("/proxy/openai/"):
            provider = "openai"
            target_path = req_path[len("/proxy/openai"):]
            upstream_base = self._default_base("openai", "https://api.openai.com")
        elif lower_path.startswith("/proxy/gemini/"):
            provider = "gemini"
            target_path = req_path[len("/proxy/gemini"):]
            upstream_base = self._default_base("gemini", "https://generativelanguage.googleapis.com")
        elif lower_path.startswith("/v1/messages") or lower_path.startswith("/messages"):
            provider = "anthropic"
            if lower_path.startswith("/messages"):
                target_path = "/v1" + target_path
            upstream_base = self._default_base("anthropic", "https://api.anthropic.com")
        elif lower_path.startswith("/v1/chat/completions") or lower_path.startswith("/chat/completions"):
            provider = "openai"
            if lower_path.startswith("/chat/completions"):
                target_path = "/v1" + target_path
            upstream_base = self._default_base("openai", "https://api.openai.com")
        elif lower_path.startswith("/v1beta/"):
            provider = "gemini"
            upstream_base = self._default_base("gemini", "https://generativelanguage.googleapis.com")
        else:
            provider = "openai"
            upstream_base = self._default_base("openai", "https://api.openai.com")

        # Auto-detect format from custom upstream URL
        if not custom_provider and custom_upstream:
            lu = custom_upstream.lower()
            if "/messages" in lu:
                provider = "anthropic"
            elif "/chat/completions" in lu or "/completions" in lu:
                provider = "openai"
            elif ":generatecontent" in lu or "/v1beta/" in lu:
                provider = "gemini"

        if custom_provider:
            provider = custom_provider

        if custom_upstream:
            parsed = urlparse(custom_upstream)
            if parsed.scheme and parsed.netloc and is_full_endpoint(parsed.path):
                return RouteInfo(
                    target_url=custom_upstream,
                    provider=provider,
                    is_custom=True,
                )
            base = custom_upstream.rstrip("/")
            return RouteInfo(
                target_url=fmt_target_url(base, target_path, forward_query),
                provider=provider,
                is_custom=True,
            )

        return RouteInfo(
            target_url=fmt_target_url(upstream_base, target_path, forward_query),
            provider=provider,
            is_custom=False,
        )

    def _default_base(self, key: str, fallback: str) -> str:
        val = self.cfg.upstream_defaults.get(key)
        if val:
            return val.rstrip("/")
        return fallback


def fmt_target_url(base: str, path: str, query: str) -> str:
    base = base.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    if base.endswith("/v1") and path.startswith("/v1/"):
        path = path[len("/v1"):]
    if query:
        return f"{base}{path}?{query}"
    return f"{base}{path}"


def is_full_endpoint(path: str) -> bool:
    lp = path.lower()
    return any(
        ep in lp
        for ep in (
            "/messages",
            "/chat/completions",
            "/completions",
            ":generatecontent",
            ":streamgeneratecontent",
        )
    )
