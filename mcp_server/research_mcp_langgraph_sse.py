# research_agent.py
from dotenv import load_dotenv
load_dotenv()

import os
import json
import subprocess
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import httpx
import sseclient
from io import StringIO

memory = ConversationBufferMemory(return_messages=True)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY")
)

# SSE-based MCP tool call

def call_mcp_tool(tool_name: str, args: dict) -> str:
    mcp_url = "http://127.0.0.1:6274/sse"
    payload = {"tool": tool_name, "args": args}

    with httpx.Client(timeout=None) as client:
        with client.stream("POST", mcp_url, json=payload, headers={"Accept": "text/event-stream"}) as response:
            if response.status_code != 200:
                raise RuntimeError(f"MCP SSE connection failed: {response.status_code} - {response.text}")

            client = sseclient.SSEClient(response)
            collected = []

            for event in client.events():
                print(f"Received event: {event.event} -> {event.data}")  # Debug line

                if event.event == "tool_result":
                    collected.append(event.data)
                elif event.event == "done":
                    break

            return "\n".join(collected).strip()

# Node 1: Planning
def plan_node(state: dict) -> dict:
    query = state.get("query", "")
    past_messages = memory.load_memory_variables({})["history"]
    system_prompt = """You are a AI research assistant. Break the user query into sub-parts to find:
    1. A summary of the topic, use web_search tool and fetch_url_content tool 
    2. Suggested reference books and author names, use web_search tool and fetch_url_content tool
    3. Related articles or papers with author names, use web_search tool and fetch_url_content tool
    Be structured and concise. Use the tools provided to fetch detailed information.
    """

    messages = [
        SystemMessage(content=system_prompt),
        *past_messages,
        HumanMessage(content=query)
    ]
    plan_response = llm.invoke(messages)
    memory.save_context({"input": query}, {"output": plan_response.content})
    return {**state, "search_plan": plan_response.content}

# Node 2: Use MCP's web_search tool
def search_node(state: dict) -> dict:
    plan = state.get("search_plan", "")
    try:
        raw_results = call_mcp_tool("web_search", {"query": plan})
        search_summary = json.loads(raw_results) if raw_results.startswith("[") else raw_results
    except Exception as e:
        search_summary = f"Failed to get search results: {str(e)}"
    return {**state, "search_results": search_summary}

# Node 3: Summarize
def summarize_node(state: dict) -> dict:
    past_messages = memory.load_memory_variables({})["history"]
    results = state.get("search_results", "")
    system_prompt = """You are a highly capable research AI. Given a topic, return:
    1. Summary of the topic (bullet points)
    2. Recommended books (bullet points)
    3. Relevant links/articles (bullet points)
    Be structured and concise."""

    messages = [
        SystemMessage(content=system_prompt),
        *past_messages,
        HumanMessage(content=str(results))
    ]
    response = llm.invoke(messages)
    memory.save_context({"input": str(results)}, {"output": response.content})
    return {**state, "final_answer": response.content}

# LangGraph flow
graph = StateGraph(dict)
graph.add_node("plan", plan_node)
graph.add_node("search", search_node)
graph.add_node("summarize", summarize_node)

graph.set_entry_point("plan")
graph.add_edge("plan", "search")
graph.add_edge("search", "summarize")
graph.add_edge("summarize", END)

runnable = graph.compile()

if __name__ == "__main__":
    user_input = input("Enter a topic to research: ")
    result = runnable.invoke({"query": user_input})
    print("\nResearch Output:\n")
    print(result["final_answer"].strip())
