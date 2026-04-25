from langchain_core.tools import BaseTool


class ToolRegistry:
    """Central registry for all available agent tools.

    Acts as the single source of truth for tool discovery, metadata,
    and binding to LLM instances.
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list(self) -> list[BaseTool]:
        return list(self._tools.values())

    def bind(self, llm) -> BaseTool:
        """Bind all registered tools to an LLM for tool-calling."""
        return llm.bind_tools(self.list())


_global_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    return _global_registry
