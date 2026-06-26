from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    """State of the agent."""

    messages: Annotated[list, add_messages]
    schema: str
    sql_query: str
    retry_count: int
    query_result: str