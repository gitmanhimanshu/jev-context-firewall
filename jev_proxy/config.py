import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Config:
    port: int = 8192
    jev_base_url: str = "https://api.codiv.ai"
    jev_api_key: str = ""
    jev_api_keys: List[str] = field(default_factory=list)
    jev_concurrency_per_key: int = 3
    jev_model: str = "openjev-latest"
    jev_timeout_ms: int = 15000
    relevance_threshold: float = 0.60
    min_confidence_threshold: float = 0.60
    tail_size: int = 4
    sacrosanct_tail_size: int = 2
    chunk_max_tokens: int = 1000
    chunk_overlap_percent: float = 0.20
    upstream_defaults: Dict[str, str] = field(
        default_factory=lambda: {
            "openai": "https://api.openai.com",
            "anthropic": "https://api.anthropic.com",
            "gemini": "https://generativelanguage.googleapis.com",
        }
    )
    upstream_headers: Dict[str, str] = field(
        default_factory=lambda: {
            "Origin": "https://trae.ai",
            "Referer": "https://trae.ai/",
            "HTTP-Referer": "https://trae.ai",
            "X-Title": "Trae",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Trae/1.0.10282 Chrome/120.0.6099.291 "
                "Electron/28.2.1 Safari/537.36 Trae-Agent"
            ),
        }
    )
    _cached_keys: Optional[List[str]] = field(default=None, repr=False)

    def get_api_keys(self) -> List[str]:
        if self._cached_keys is not None:
            return self._cached_keys

        raw_keys: List[str] = []
        for k in self.jev_api_keys:
            raw_keys.extend(parse_keys_from_env(k))

        if self.jev_api_key:
            raw_keys.extend(parse_keys_from_env(self.jev_api_key))

        seen = set()
        unique = []
        for k in raw_keys:
            if k and k not in seen:
                seen.add(k)
                unique.append(k)

        self._cached_keys = unique
        return unique

    def get_concurrency(self) -> int:
        keys = self.get_api_keys()
        per_key = self.jev_concurrency_per_key if self.jev_concurrency_per_key > 0 else 3
        if not keys:
            return per_key
        return len(keys) * per_key


def parse_keys_from_env(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []

    # Support JSON array format: ["k1", "k2"]
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                cleaned = [str(k).strip("\"' \t\r\n") for k in parsed if str(k).strip("\"' \t\r\n")]
                if cleaned:
                    return cleaned
        except Exception:
            pass

    # Split by comma, newline, semicolon, tab, or space
    parts = re.split(r"[,;\n\r\t ]+", raw)
    result = []
    for p in parts:
        trimmed = p.strip("\"' \t\r\n")
        if trimmed:
            result.append(trimmed)
    return result


def load_dotenv(filepath: str = ".env") -> None:
    target = filepath
    if not os.path.exists(target):
        target = os.path.join("..", filepath)
        if not os.path.exists(target):
            return

    try:
        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("=", 1)
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                val = parts[1].strip()
                if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
                    val = val[1:-1]
                if key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


def load_config(path: str = "config.json") -> Config:
    load_dotenv(".env")
    cfg = Config()

    # Read config.json if present
    target_path = path
    if not os.path.exists(target_path):
        alt = os.path.join("..", path)
        if os.path.exists(alt):
            target_path = alt

    if os.path.exists(target_path):
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
        except Exception as e:
            print(f"[WARN] Error parsing {target_path}: {e}")

    # Override config fields from environment variables
    if os.getenv("JEV_API_KEYS"):
        cfg.jev_api_keys.extend(parse_keys_from_env(os.getenv("JEV_API_KEYS", "")))
    if os.getenv("JEV_API_KEY"):
        cfg.jev_api_key = os.getenv("JEV_API_KEY", "")

    # Indexed keys for cloud platforms (e.g. JEV_API_KEY_1 .. 30)
    for i in range(1, 31):
        kname = f"JEV_API_KEY_{i}"
        val = os.getenv(kname)
        if val:
            cfg.jev_api_keys.extend(parse_keys_from_env(val))

    if os.getenv("JEV_CONCURRENCY_PER_KEY"):
        try:
            cfg.jev_concurrency_per_key = int(os.getenv("JEV_CONCURRENCY_PER_KEY", "3"))
        except ValueError:
            pass

    if os.getenv("JEV_BASE_URL"):
        cfg.jev_base_url = os.getenv("JEV_BASE_URL", "").rstrip("/")

    if os.getenv("JEV_MODEL"):
        cfg.jev_model = os.getenv("JEV_MODEL", "")

    if os.getenv("PORT"):
        try:
            cfg.port = int(os.getenv("PORT", "8192"))
        except ValueError:
            pass

    if os.getenv("UPSTREAM_ANTHROPIC"):
        cfg.upstream_defaults["anthropic"] = os.getenv("UPSTREAM_ANTHROPIC", "").rstrip("/")
    if os.getenv("UPSTREAM_OPENAI"):
        cfg.upstream_defaults["openai"] = os.getenv("UPSTREAM_OPENAI", "").rstrip("/")
    if os.getenv("UPSTREAM_GEMINI"):
        cfg.upstream_defaults["gemini"] = os.getenv("UPSTREAM_GEMINI", "").rstrip("/")

    return cfg
