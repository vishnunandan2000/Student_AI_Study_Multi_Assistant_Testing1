from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from backend.agents.state import AgentState
from backend.core.llm import get_llm
import json
import logging

logger = logging.getLogger(__name__)

SUPPORTED_AGENTS = {
    "study_agent",
    "exam_agent",
    "coding_agent",
    "project_agent",
    "document_agent",
    "audio_agent",
    "video_agent",
    "memory_agent",
    "personalization_agent",
}

def router_node(state: AgentState):
    """Orchestrator to route to the correct agent based on the task"""
    llm = get_llm()
    messages = state.get("messages", [])
    
    system_prompt = """You are the Orchestrator Agent for an AI Student Assistant.
Your job is to analyze the user's request and determine which specialized agent should handle it.
Available Agents:
- study_agent: Explains concepts, creates study plans, generates notes/summaries.
- exam_agent: Creates quizzes, mock tests, important questions.
- coding_agent: Debugs code, explains code, DSA trainer.
- project_agent: Generates project ideas, architectures, documentation.
- document_agent: Analyzes uploaded documents, text files, and notes.
- audio_agent: Text to speech and audio learning assistant.
- video_agent: Educational video script and explanation generator.
- memory_agent: Flashcards, memory techniques, spaced repetition tracking.
- personalization_agent: Progress tracking, weak topics, personalized schedules.

Respond with ONLY a JSON object containing the `next_agent` key. Example: {"next_agent": "study_agent"}"""
    
    try:
        response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        elif "{" in content and "}" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            content = content[start:end].strip()
        
        data = json.loads(content)
        next_agent = data.get("next_agent", "study_agent")
        if next_agent not in SUPPORTED_AGENTS:
            next_agent = "study_agent"
    except Exception as e:
        logger.warning(f"Router parsing failed ({e}). Defaulting to 'study_agent'")
        next_agent = "study_agent"
        
    return {"next_agent": next_agent}

def study_node(state: AgentState):
    llm = get_llm()
    system_prompt = "You are an expert AI Teacher. Explain concepts simply, step-by-step with real-world examples."
    response = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response], "generated_content": {"type": "study", "content": response.content}}

def exam_node(state: AgentState):
    llm = get_llm()
    system_prompt = "You are an Exam Coach. Create quizzes, predict important questions, and prepare mock tests."
    response = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response], "generated_content": {"type": "exam", "content": response.content}}

def coding_node(state: AgentState):
    llm = get_llm()
    system_prompt = "You are a Senior Coding Mentor. Debug code, explain step-by-step, and optimize solutions."
    response = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response], "generated_content": {"type": "coding", "content": response.content}}

def project_node(state: AgentState):
    llm = get_llm()
    system_prompt = "You are a Project Guide. Help design architecture, create module breakdowns, and documentation."
    response = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response], "generated_content": {"type": "project", "content": response.content}}

def document_node(state: AgentState):
    llm = get_llm()
    system_prompt = "You are a Document Analysis Specialist. Summarize, extract key takeaways, and answer queries about documents."
    response = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response], "generated_content": {"type": "document", "content": response.content}}

def audio_node(state: AgentState):
    llm = get_llm()
    system_prompt = "You are an Audio Learning Assistant. Format explanations into clear spoken-word scripts."
    response = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response], "generated_content": {"type": "audio", "content": response.content}}

def video_node(state: AgentState):
    llm = get_llm()
    system_prompt = "You are a Video Script Creator. Turn explanations into structured video scripts with visual cues."
    response = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response], "generated_content": {"type": "video", "content": response.content}}

def memory_node(state: AgentState):
    llm = get_llm()
    system_prompt = "You are a Memory & Flashcard Expert. Create spaced repetition flashcards and mnemonics."
    response = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response], "generated_content": {"type": "memory", "content": response.content}}

def personalization_node(state: AgentState):
    llm = get_llm()
    system_prompt = "You are a Personalized Learning Coach. Track learning progress and recommend tailored study schedules."
    response = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response], "generated_content": {"type": "personalization", "content": response.content}}

# Build Graph
def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("orchestrator", router_node)
    workflow.add_node("study_agent", study_node)
    workflow.add_node("exam_agent", exam_node)
    workflow.add_node("coding_agent", coding_node)
    workflow.add_node("project_agent", project_node)
    workflow.add_node("document_agent", document_node)
    workflow.add_node("audio_agent", audio_node)
    workflow.add_node("video_agent", video_node)
    workflow.add_node("memory_agent", memory_node)
    workflow.add_node("personalization_agent", personalization_node)
    
    # Set entry point
    workflow.set_entry_point("orchestrator")
    
    # Conditional Edges
    def route_to_agent(state: AgentState):
        agent = state.get("next_agent", "study_agent")
        return agent if agent in SUPPORTED_AGENTS else "study_agent"
        
    agent_map = {agent: agent for agent in SUPPORTED_AGENTS}
    workflow.add_conditional_edges("orchestrator", route_to_agent, agent_map)
    
    # End edges for all nodes
    for agent in SUPPORTED_AGENTS:
        workflow.add_edge(agent, END)
        
    return workflow.compile()

app_graph = build_graph()

