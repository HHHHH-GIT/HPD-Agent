from langgraph.graph import StateGraph, END

from src.core.enums import TaskDifficulty
from src.core.state import AgentState
from src.nodes import first_level_assessment, direct_answer, scheduler_node, synthesizer


def _route_after_assessment(state: AgentState) -> str:
    """Conditional edge: simple → direct_answer, complex → coordinator."""
    difficulty = state.get("analysis")
    if difficulty == TaskDifficulty.SIMPLE:
        return "direct_answer"
    return "coordinator"


def build_graph() -> StateGraph:
    """Assemble the agent graph.

    Simple path:       assessment → direct_answer → END
    Complex path (1): assessment → coordinator (planning only)
    Complex path (2): coordinator → scheduler_node (Kahn parallel execution)
    Complex path (3): scheduler_node → synthesizer (streaming synthesis)

    Each node has a single, clear responsibility:
      - coordinator:  LLM planning — decompose into DAG
      - scheduler:    Kahn orchestration — execute sub-tasks in parallel
      - synthesizer:  Build streaming prompt from all results
    """
    from src.agents import coordinator

    graph = StateGraph(AgentState)

    graph.add_node("first_level_assessment", first_level_assessment)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("coordinator", coordinator)
    graph.add_node("scheduler_node", scheduler_node)
    graph.add_node("synthesizer", synthesizer)

    graph.set_entry_point("first_level_assessment")

    graph.add_conditional_edges(
        source="first_level_assessment",
        path=_route_after_assessment,
        path_map={"direct_answer": "direct_answer", "coordinator": "coordinator"},
    )

    graph.add_edge("direct_answer", END)
    graph.add_edge("coordinator", "scheduler_node")
    graph.add_edge("scheduler_node", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph
