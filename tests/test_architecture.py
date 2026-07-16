"""Guardrails for dependency direction and small functions."""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1]
IGNORED_PARTS = {".git", ".venv", "__pycache__"}


def python_files() -> list[Path]:
    paths = [
        path for path in ROOT.rglob("*.py")
        if not IGNORED_PARTS.intersection(path.parts)
    ]
    return paths


def oversized_functions(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = (
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    nodes = [
        f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}"
        for node in functions if node.end_lineno - node.lineno + 1 > 15
    ]
    return nodes


def test_functions_are_at_most_fifteen_lines() -> None:
    violations = [
        item for path in python_files()
        for item in oversized_functions(path)
    ]
    assert violations == []


def test_api_does_not_import_analysis_implementation() -> None:
    source = (ROOT / "api/main.py").read_text(encoding="utf-8")
    forbidden = ("from services.", "from agents.", "from asr.", "from utils.")
    assert not any(module in source for module in forbidden)
