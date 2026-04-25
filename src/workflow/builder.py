from langgraph.graph import StateGraph, END

from src.core.enums import TaskDifficulty
from src.core.state import AgentState
from src.nodes import (
    first_level_assessment,
    direct_answer,
    decomposer,
    scheduler_node,
    synthesizer,
)


def _route_after_assessment(state: AgentState) -> str:
    """Conditional edge: simple → direct_answer, complex → decomposer."""
    difficulty = state.get("analysis")
    if difficulty == TaskDifficulty.SIMPLE:
        return "direct_answer"
    return "decomposer"


def build_graph() -> StateGraph:
    """Assemble the agent graph.

    Simple path:  assessment → direct_answer → END
    Complex path: assessment → decomposer → scheduler → synthesizer → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("first_level_assessment", first_level_assessment)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("decomposer", decomposer)
    graph.add_node("scheduler", scheduler_node)
    graph.add_node("synthesizer", synthesizer)

    graph.set_entry_point("first_level_assessment")

    graph.add_conditional_edges(
        source="first_level_assessment",
        path=_route_after_assessment,
        path_map={"direct_answer": "direct_answer", "decomposer": "decomposer"},
    )

    graph.add_edge("direct_answer", END)
    graph.add_edge("decomposer", "scheduler")
    graph.add_edge("scheduler", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph
