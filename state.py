from typing import TypedDict, List

class GraphState(TypedDict):
    original_query: str
    current_query: str
    retrieved_context: str
    num_refinement: int
    num_planning: int
    next_node: str
    feedback_history: List[str]
    final_answer: str

