"""Tree-sitter parsing helpers for syntax outline extraction."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, cast

from tree_sitter import Language, Node, Parser, Query, QueryCursor
from tree_sitter_language_pack import get_language

from src.code_intel.core import ProviderUnavailable, Range, Symbol, SymbolKind
from src.code_intel.core.languages import SUPPORTED_CODE_LANGUAGES, language_for_path
from src.code_intel.core.models import validate_workspace_relative_path

TREE_SITTER_QUERY_VERSION = "tree-sitter-query-v1"
TREE_SITTER_INDEX_VERSION = f"tree-sitter:{TREE_SITTER_QUERY_VERSION}"
TREE_SITTER_SOURCE = "tree_sitter"
TREE_SITTER_CONFIDENCE = 0.72
SUPPORTED_LANGUAGES = SUPPORTED_CODE_LANGUAGES

_CAPTURE_KINDS = {
    "class.definition": SymbolKind.CLASS,
    "function.definition": SymbolKind.FUNCTION,
    "method.definition": SymbolKind.METHOD,
    "interface.definition": SymbolKind.INTERFACE,
    "import.definition": SymbolKind.IMPORT,
    "export.definition": SymbolKind.EXPORT,
}
_PARENT_KIND_ALLOWLIST = {
    SymbolKind.MODULE,
    SymbolKind.CLASS,
    SymbolKind.INTERFACE,
    SymbolKind.FUNCTION,
    SymbolKind.METHOD,
}

LanguageLoader = Callable[[str], object]
ParserLoader = Callable[[str], object]
_DEFAULT_LANGUAGE_LOADER = cast(LanguageLoader, get_language)


def _default_parser_loader(language: str) -> Parser:
    parser = Parser()
    parser.language = cast(Language, get_language(language))
    return parser


_DEFAULT_PARSER_LOADER = cast(ParserLoader, _default_parser_loader)


class TreeSitterGrammarUnavailable(ProviderUnavailable):
    """Grammar loading failure exposed as a safe Chinese ToolError."""

    message: ClassVar[str] = "Tree-sitter 语法包不可用。"
    hint: ClassVar[str | None] = (
        "请安装或修复 tree-sitter-language-pack："
        "python3 -m pip install tree-sitter-language-pack，并确认包含 Python/TypeScript/JavaScript grammar。"
    )


@dataclass(frozen=True)
class _SymbolDraft:
    kind: SymbolKind
    node: Node
    name_node: Node | None
    name: str
    signature: str | None

    @property
    def sort_key(self) -> tuple[int, int, int, int, str, str]:
        start = _point_tuple(_node_position(self.node, "start_position"))
        end = _point_tuple(_node_position(self.node, "end_position"))
        return (start[0], start[1], end[0], end[1], self.kind.value, self.name)


@dataclass(frozen=True)
class _SymbolRecord:
    symbol: Symbol
    node: Node | None


class TreeSitterParser:
    """Load grammars, run queries, and normalize captures into core Symbols."""

    def __init__(
        self,
        *,
        query_dir: Path | None = None,
        language_loader: LanguageLoader = _DEFAULT_LANGUAGE_LOADER,
        parser_loader: ParserLoader = _DEFAULT_PARSER_LOADER,
    ) -> None:
        self.query_dir: Path = query_dir or Path(__file__).with_name("queries")
        self._language_loader: LanguageLoader = language_loader
        self._parser_loader: ParserLoader = parser_loader
        self._languages: dict[str, Language] = {}
        self._queries: dict[str, Query] = {}

    def extract_symbols(
        self, workspace_root: Path, path: str, language: str | None = None
    ) -> list[Symbol]:
        """Parse one workspace-relative file and return deterministic Symbol models."""
        relative_path = validate_workspace_relative_path(path)
        resolved_language = self.resolve_language(relative_path, language)
        absolute_path = self._resolve_path(workspace_root, relative_path)
        raw_content = absolute_path.read_bytes()
        file_hash = hashlib.sha256(raw_content).hexdigest()[:16]
        parser = self._load_parser(resolved_language)
        tree = parser.parse(raw_content)
        root_node = tree.root_node
        root = root_node() if callable(root_node) else root_node
        module_record = self._module_record(
            root, relative_path, resolved_language, file_hash
        )
        records = [module_record]

        drafts = sorted(
            self._drafts_from_query(resolved_language, root, raw_content),
            key=lambda draft: draft.sort_key,
        )
        for draft in drafts:
            kind = self._kind_for_draft(draft)
            parent = self._nearest_parent_record(records, draft.node)
            qualified_name = self._qualified_name(draft.name, parent)
            symbol = Symbol(
                name=draft.name,
                qualified_name=qualified_name,
                kind=kind,
                language=resolved_language,
                path=relative_path,
                range=_range_for_node(draft.node),
                selection_range=_range_for_node(draft.name_node or draft.node),
                parent_id=parent.symbol.id if parent is not None else None,
                signature=draft.signature,
                doc=None,
                source=TREE_SITTER_SOURCE,
                confidence=TREE_SITTER_CONFIDENCE,
                file_hash=file_hash,
                index_version=TREE_SITTER_INDEX_VERSION,
            )
            records.append(_SymbolRecord(symbol=symbol, node=draft.node))

        return [record.symbol for record in records]

    @staticmethod
    def resolve_language(path: str, requested_language: str | None = None) -> str:
        """Resolve a supported provider language from an explicit language or extension."""
        if requested_language is not None:
            normalized = requested_language.casefold()
            if normalized in SUPPORTED_LANGUAGES:
                return normalized
            raise TreeSitterGrammarUnavailable(
                f"unsupported language: {requested_language}"
            )
        suffix = Path(path).suffix.casefold()
        language = language_for_path(path)
        if language is None:
            raise TreeSitterGrammarUnavailable(
                f"unsupported path extension: {suffix or '<none>'}"
            )
        return language

    def _resolve_path(self, workspace_root: Path, relative_path: str) -> Path:
        absolute_path = (workspace_root / relative_path).resolve(strict=False)
        try:
            _ = absolute_path.relative_to(workspace_root)
        except ValueError as error:
            raise TreeSitterGrammarUnavailable("path escaped workspace") from error
        if not absolute_path.is_file():
            raise TreeSitterGrammarUnavailable(f"file not found: {relative_path}")
        return absolute_path

    def _load_language(self, language: str) -> Language:
        existing = self._languages.get(language)
        if existing is not None:
            return existing
        try:
            loaded_language = cast(Language, self._language_loader(language))
        except Exception as error:
            raise TreeSitterGrammarUnavailable(
                f"missing grammar for {language}"
            ) from error
        self._languages[language] = loaded_language
        return loaded_language

    def _load_parser(self, language: str) -> Parser:
        try:
            loaded_parser = cast(Parser, self._parser_loader(language))
            return loaded_parser
        except Exception as error:
            raise TreeSitterGrammarUnavailable(
                f"missing parser for {language}"
            ) from error

    def _load_query(self, language: str) -> Query:
        existing = self._queries.get(language)
        if existing is not None:
            return existing
        language_object = self._load_language(language)
        query_path = self.query_dir / f"{language}.scm"
        try:
            query_source = query_path.read_text(encoding="utf-8")
            query = Query(language_object, query_source)
        except Exception as error:
            raise TreeSitterGrammarUnavailable(
                f"query unavailable for {language}"
            ) from error
        self._queries[language] = query
        return query

    def _drafts_from_query(
        self, language: str, root: Node, raw_content: bytes
    ) -> list[_SymbolDraft]:
        query = self._load_query(language)
        drafts: list[_SymbolDraft] = []
        seen: set[tuple[str, int, int, str]] = set()
        for _pattern_index, captures in QueryCursor(query).matches(root):
            name_node = self._first_capture(captures, "name")
            for capture_name, kind in _CAPTURE_KINDS.items():
                for node in captures.get(capture_name, []):
                    name = self._name_for_node(kind, node, name_node, raw_content)
                    key = (
                        kind.value,
                        _node_byte(node, "start_byte"),
                        _node_byte(node, "end_byte"),
                        name,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    drafts.append(
                        _SymbolDraft(
                            kind=kind,
                            node=node,
                            name_node=(
                                name_node
                                if self._node_contains(node, name_node)
                                else None
                            ),
                            name=name,
                            signature=self._signature_for_node(node, raw_content),
                        )
                    )
        return drafts

    @staticmethod
    def _first_capture(captures: dict[str, list[Node]], name: str) -> Node | None:
        nodes = captures.get(name)
        if not nodes:
            return None
        return nodes[0]

    def _name_for_node(
        self, kind: SymbolKind, node: Node, name_node: Node | None, raw_content: bytes
    ) -> str:
        if name_node is not None and self._node_contains(node, name_node):
            return _node_text(name_node, raw_content)
        field_name = _node_child_by_field_name(node, "name")
        if field_name is not None:
            return _node_text(field_name, raw_content)
        if kind == SymbolKind.IMPORT:
            return _first_statement_line(node, raw_content)
        if kind == SymbolKind.EXPORT:
            return self._export_name(node, raw_content)
        return _first_statement_line(node, raw_content)

    def _export_name(self, node: Node, raw_content: bytes) -> str:
        text = _first_statement_line(node, raw_content)
        discovered_name = self._first_descendant_name(node, raw_content)
        if discovered_name is not None:
            return f"export {discovered_name}"
        if text.startswith("export {") and "}" in text:
            exported = text.removeprefix("export {").split("}", 1)[0].strip()
            first_export = exported.split(",", 1)[0].strip()
            if first_export:
                return f"export {first_export}"
        return text.rstrip(";")

    def _first_descendant_name(self, node: Node, raw_content: bytes) -> str | None:
        stack = _named_children(node)
        while stack:
            current = stack.pop(0)
            field_name = _node_child_by_field_name(current, "name")
            if field_name is not None:
                return _node_text(field_name, raw_content)
            stack.extend(_named_children(current))
        return None

    def _kind_for_draft(self, draft: _SymbolDraft) -> SymbolKind:
        if draft.kind != SymbolKind.FUNCTION:
            return draft.kind
        if self._is_python_or_javascript_method(draft.node):
            return SymbolKind.METHOD
        return draft.kind

    @staticmethod
    def _is_python_or_javascript_method(node: Node) -> bool:
        current = _node_parent(node)
        found_function_parent = False
        while current is not None:
            if _node_kind(current) in {
                "function_definition",
                "function_declaration",
                "method_definition",
            }:
                found_function_parent = True
            if _node_kind(current) in {"class_definition", "class_declaration"}:
                return not found_function_parent
            current = _node_parent(current)
        return False

    def _module_record(
        self, root: Node, path: str, language: str, file_hash: str
    ) -> _SymbolRecord:
        module_name = Path(path).stem
        module_symbol = Symbol(
            name=module_name,
            qualified_name=module_name,
            kind=SymbolKind.MODULE,
            language=language,
            path=path,
            range=_range_for_node(root),
            selection_range=_range_for_node(root),
            parent_id=None,
            signature=None,
            doc=None,
            source=TREE_SITTER_SOURCE,
            confidence=TREE_SITTER_CONFIDENCE,
            file_hash=file_hash,
            index_version=TREE_SITTER_INDEX_VERSION,
        )
        return _SymbolRecord(symbol=module_symbol, node=root)

    def _nearest_parent_record(
        self, records: list[_SymbolRecord], node: Node
    ) -> _SymbolRecord | None:
        candidates = [
            record
            for record in records
            if record.node is not None
            and record.symbol.kind in _PARENT_KIND_ALLOWLIST
            and record.node != node
            and self._node_contains(record.node, node)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda record: (
                _node_byte(record.node, "end_byte") - _node_byte(record.node, "start_byte")
                if record.node
                else 0
            ),
        )

    @staticmethod
    def _qualified_name(name: str, parent: _SymbolRecord | None) -> str:
        if parent is None or parent.symbol.kind == SymbolKind.MODULE:
            return name
        parent_name = parent.symbol.qualified_name or parent.symbol.name
        return f"{parent_name}.{name}"

    @staticmethod
    def _node_contains(outer: Node, inner: Node | None) -> bool:
        if inner is None:
            return False
        return _node_byte(outer, "start_byte") <= _node_byte(inner, "start_byte") and _node_byte(
            inner, "end_byte"
        ) <= _node_byte(outer, "end_byte")

    @staticmethod
    def _signature_for_node(node: Node, raw_content: bytes) -> str | None:
        if _node_kind(node) in {"module", "program"}:
            return None
        return _first_statement_line(node, raw_content)


def _point_tuple(point: object) -> tuple[int, int]:
    row = getattr(point, "row", None)
    column = getattr(point, "column", None)
    if isinstance(row, int) and isinstance(column, int):
        return (row, column)
    pair = cast(Sequence[int], point)
    return (int(pair[0]), int(pair[1]))


def _range_for_node(node: Node) -> Range:
    start_line, start_col = _point_tuple(_node_position(node, "start_position"))
    end_line, end_col = _point_tuple(_node_position(node, "end_position"))
    return Range(
        start_line=start_line, start_col=start_col, end_line=end_line, end_col=end_col
    )


def _node_text(node: Node, raw_content: bytes) -> str:
    return raw_content[_node_byte(node, "start_byte") : _node_byte(node, "end_byte")].decode(
        "utf-8", errors="replace"
    )


def _first_statement_line(node: Node, raw_content: bytes) -> str:
    text = _node_text(node, raw_content).strip()
    if not text:
        return _node_kind(node)
    return text.splitlines()[0].strip()


def _node_value(node: Node, name: str) -> object:
    attr = getattr(node, name)
    return attr() if callable(attr) else attr


def _node_kind(node: Node) -> str:
    attr_name = "kind" if hasattr(node, "kind") else "type"
    kind = _node_value(node, attr_name)
    return str(kind)


def _node_byte(node: Node, name: str) -> int:
    value = _node_value(node, name)
    return int(value)


def _node_position(node: Node, name: str) -> object:
    attr_name = name
    if not hasattr(node, attr_name):
        attr_name = "start_point" if name == "start_position" else "end_point"
    return _node_value(node, attr_name)


def _node_parent(node: Node) -> Node | None:
    parent = _node_value(node, "parent")
    return cast(Node | None, parent)


def _node_child_by_field_name(node: Node, field_name: str) -> Node | None:
    getter = getattr(node, "child_by_field_name")
    child = getter(field_name) if callable(getter) else None
    return cast(Node | None, child)


def _named_children(node: Node) -> list[Node]:
    existing = getattr(node, "named_children", None)
    if existing is not None:
        return list(cast(Sequence[Node], existing))

    children: list[Node] = []
    count_value = getattr(node, "named_child_count")
    count = int(count_value() if callable(count_value) else count_value)
    child_getter = getattr(node, "named_child")
    for index in range(count):
        child = child_getter(index) if callable(child_getter) else None
        children.append(cast(Node, child))
    return children


__all__ = [
    "SUPPORTED_LANGUAGES",
    "TREE_SITTER_CONFIDENCE",
    "TREE_SITTER_INDEX_VERSION",
    "TREE_SITTER_QUERY_VERSION",
    "TREE_SITTER_SOURCE",
    "TreeSitterGrammarUnavailable",
    "TreeSitterParser",
]
