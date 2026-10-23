from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    task_type: str # e.g., 'study', 'exam', 'coding', 'project', 'document', 'audio', 'video'
    metadata: Dict[str, Any]
    next_agent: str
    errors: List[str]
    current_topic: str
    learning_history: List[Dict[str, Any]]
    generated_content: Dict[str, Any] # stores output from specific agents
