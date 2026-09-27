import re
from dataclasses import dataclass, field
from typing import List, Set

FILE_PATH_REGEX = re.compile(
    r"(?i)\b([a-zA-Z0-9_\-\./]+\.(?:ts|tsx|js|jsx|go|py|rs|java|cpp|c|h|cs|php|rb|json|yaml|yml|sql|html|css|env|toml|md))\b"
)
FUNCTION_REGEX = re.compile(r"\b([a-zA-Z0-9_]{3,})\s*\(")
ERROR_REGEX = re.compile(
    r"(?i)\b(TypeError|ReferenceError|SyntaxError|Error|Exception|403|401|404|500|502|failed|panic|timeout)\b"
)
ANAPHORA_REGEX = re.compile(
    r"(?i)\b(that|it|this|the previous|the earlier|undo|revert|same thing|again)\b"
)

# Sensitive secrets regex for scrubbing before Jev evaluation
BEARER_REGEX = re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9_\-\.]{20,}")
API_KEY_REGEX = re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*[\"']?[A-Za-z0-9_\-\.]{16,}[\"']?")
URI_AUTH_REGEX = re.compile(r"(?i)(postgres|mysql|mongodb|redis):\/\/[^:]+:[^@]+@")

# IDE prompt reminders regex
SYSTEM_REMINDER_REGEX = re.compile(r"(?s)<system-reminder>.*?</system-reminder>")

STOP_SYMBOLS = {
    "main", "new", "len", "make", "close", "print", "println", "string", "int", "error",
    "context", "json", "with", "set", "get", "parse", "read", "write", "exec", "run",
    "init", "start", "stop", "test", "import", "func", "var", "const", "type", "append",
    "delete", "panic", "recover", "format", "log", "info", "debug", "warn", "err", "data",
    "req", "res", "resp", "ctx", "done", "load", "handle", "handler", "routes", "call",
    "send", "recv", "time", "date", "byte", "dumps", "loads", "items", "keys", "values",
    "range", "strip", "split", "join", "encode", "decode", "count", "agent", "agents", "model",
    "models", "dict", "list", "none", "true", "false", "self", "class", "search", "query",
    "update", "select", "create", "open", "title",
}

GENERIC_FILES = {
    "main.go", "index.ts", "index.js", "app.ts", "app.js", "init.go", "types.go", "models.go",
    "db.go", "res.go", "env.go", "config.go", "readme.md", "package.json", "go.mod", "go.sum",
    "utils.py", "utils.ts", "utils.js", "helpers.py", "helpers.ts", "config.py", "test.py", "test.go",
    "database.py", "constants.py", "tools.py",
}


@dataclass
class EntityFootprint:
    files: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "files": self.files,
            "symbols": self.symbols,
            "errors": self.errors,
        }

    def intersects(self, other: "EntityFootprint") -> bool:
        # Check non-generic files intersection
        for f in self.files:
            base = f.lower()
            if base in GENERIC_FILES:
                continue
            is_naked = "/" not in f and "\\" not in f
            if is_naked and (len(self.files) > 5 or len(other.files) > 5):
                continue
            for of in other.files:
                if f.lower() == of.lower():
                    return True

        # Check specific symbol intersection (require at least 2 distinct non-trivial symbols)
        shared_count = 0
        for s in self.symbols:
            s_lower = s.lower()
            if s_lower in STOP_SYMBOLS or len(s) < 5:
                continue
            for os in other.symbols:
                if s_lower == os.lower():
                    shared_count += 1
                    if shared_count >= 2:
                        return True
                    break
        return False


def extract_entities(text: str) -> EntityFootprint:
    # Strip IDE boilerplate system reminders
    clean_text = SYSTEM_REMINDER_REGEX.sub("", text or "")

    files_set = set()
    symbols_set = set()
    errors_set = set()

    for m in FILE_PATH_REGEX.findall(clean_text):
        trimmed = m.strip("./ \t\r\n")
        if len(trimmed) > 3:
            files_set.add(trimmed)

    for m in FUNCTION_REGEX.findall(clean_text):
        fn = m.strip()
        if fn not in {"if", "for", "while", "switch", "return", "catch", "def", "class"}:
            symbols_set.add(fn)

    for m in ERROR_REGEX.findall(clean_text):
        errors_set.add(m.lower())

    return EntityFootprint(
        files=sorted(list(files_set)),
        symbols=sorted(list(symbols_set)),
        errors=sorted(list(errors_set)),
    )


def has_anaphora(query: str) -> bool:
    return bool(ANAPHORA_REGEX.search(query or ""))


def sanitize_secrets(text: str) -> str:
    if not text:
        return ""
    res = BEARER_REGEX.sub(r"\g<1>[REDACTED]", text)
    res = API_KEY_REGEX.sub(r"\g<1>: [REDACTED]", res)
    res = URI_AUTH_REGEX.sub(r"\g<1>://user:[REDACTED]@", res)
    return res
