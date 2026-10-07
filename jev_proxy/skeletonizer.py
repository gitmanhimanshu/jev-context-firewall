import os
import re
from typing import List, Optional

PYTHON_SIG_REGEX = re.compile(
    r"^(?:[ \t]*(?:class\s+[A-Za-z0-9_]+(?:\([^)]*\))?|async\s+def\s+[A-Za-z0-9_]+\s*\([^)]*\)(?:\s*->\s*[^:]+)?|def\s+[A-Za-z0-9_]+\s*\([^)]*\)(?:\s*->\s*[^:]+)?)):",
    re.MULTILINE,
)

GO_SIG_REGEX = re.compile(
    r"^(?:func\s+(?:\([^)]+\)\s+)?[A-Za-z0-9_]+\s*\([^)]*\)(?:\s*(?:\([^)]*\)|[A-Za-z0-9_*\[\]]+))?|type\s+[A-Za-z0-9_]+\s+(?:struct|interface))",
    re.MULTILINE,
)

TS_JS_SIG_REGEX = re.compile(
    r"^(?:[ \t]*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\s+[A-Za-z0-9_]+\s*<[^>]*>?\s*\([^)]*\)|class\s+[A-Za-z0-9_]+|interface\s+[A-Za-z0-9_]+|type\s+[A-Za-z0-9_]+\s*=|(?:const|let|var)\s+[A-Za-z0-9_]+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>))",
    re.MULTILINE,
)

RUST_SIG_REGEX = re.compile(
    r"^(?:[ \t]*(?:pub(?:\([^)]+\))?\s+)?(?:fn\s+[A-Za-z0-9_]+|struct\s+[A-Za-z0-9_]+|enum\s+[A-Za-z0-9_]+|trait\s+[A-Za-z0-9_]+|impl(?:\s+<[^>]*>)?\s+[A-Za-z0-9_]+))",
    re.MULTILINE,
)


def extract_python_skeleton(code: str) -> List[str]:
    lines = code.split("\n")
    skeleton_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped.startswith("class ")
            or stripped.startswith("def ")
            or stripped.startswith("async def ")
            or stripped.startswith("@")
        ):
            indent = len(line) - len(line.lstrip())
            skeleton_lines.append(line.rstrip())
            if stripped.endswith(":"):
                skeleton_lines.append(" " * (indent + 4) + "...")
    return skeleton_lines


def extract_go_skeleton(code: str) -> List[str]:
    lines = code.split("\n")
    skeleton_lines: List[str] = []
    in_type_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("type ") and ("struct {" in stripped or "interface {" in stripped):
            skeleton_lines.append(line)
            in_type_block = True
            continue

        if in_type_block:
            if stripped == "}":
                skeleton_lines.append(line)
                in_type_block = False
            elif stripped and not stripped.startswith("//"):
                skeleton_lines.append(line)
            continue

        if stripped.startswith("func "):
            # Retain function signature, replace body with ...
            sig = line.split("{")[0].strip()
            skeleton_lines.append(sig + " { ... }")

    return skeleton_lines


def extract_ts_js_skeleton(code: str) -> List[str]:
    lines = code.split("\n")
    skeleton_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if any(
            stripped.startswith(kw)
            for kw in (
                "export interface ",
                "interface ",
                "export type ",
                "type ",
                "export class ",
                "class ",
                "export function ",
                "export async function ",
                "function ",
                "async function ",
            )
        ):
            sig = line.split("{")[0].strip()
            if "interface " in stripped or "type " in stripped:
                skeleton_lines.append(line.rstrip())
            else:
                skeleton_lines.append(sig + " { ... }")
    return skeleton_lines


def extract_rust_skeleton(code: str) -> List[str]:
    lines = code.split("\n")
    skeleton_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if any(
            stripped.startswith(prefix)
            for prefix in (
                "pub fn ", "fn ", "pub struct ", "struct ",
                "pub enum ", "enum ", "pub trait ", "trait ", "impl "
            )
        ):
            sig = line.split("{")[0].strip()
            skeleton_lines.append(sig + " { ... }")
    return skeleton_lines


def generate_code_skeleton(code: str, filename: str, max_chars: int = 1500) -> Optional[str]:
    """
    Extracts high-signal structural outline (classes, signatures, interfaces)
    from code while discarding voluminous function bodies.
    Reduces tokens by 90-95% while keeping model aware of symbol definitions.
    """
    if not code or len(code) < 50:
        return None

    ext = os.path.splitext(filename.lower())[1]
    skeleton_lines: List[str] = []

    if ext == ".py":
        skeleton_lines = extract_python_skeleton(code)
    elif ext == ".go":
        skeleton_lines = extract_go_skeleton(code)
    elif ext in (".ts", ".tsx", ".js", ".jsx"):
        skeleton_lines = extract_ts_js_skeleton(code)
    elif ext == ".rs":
        skeleton_lines = extract_rust_skeleton(code)

    if not skeleton_lines or len(skeleton_lines) < 2:
        return None

    result = "\n".join(skeleton_lines)
    if len(result) > max_chars:
        result = result[:max_chars].rsplit("\n", 1)[0] + "\n// ... [remaining signatures truncated]"

    return result

