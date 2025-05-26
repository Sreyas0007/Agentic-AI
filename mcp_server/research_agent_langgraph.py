# research_agent.py
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage, HumanMessage
import requests

GEMINI_API_KEY = "AIzaSyCLZROEOc1W2o3cs13SnW72lueyJiBdSEU"
SERPER_API_KEY = "58c2fdf5d861def70bb30f7f6a6210fd626f4d1a"

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)

# Tool: Google Serper Search
@tool
def serper_search(query: str) -> str:
    """Search the web using Google Serper and return summarized results."""
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"q": query}
    response = requests.post(url, headers=headers, json=payload)
    return response.text

class AgentState(dict): pass

def plan_node(state: AgentState) -> AgentState:
    query = state["query"]
    system_prompt = """You are a research assistant. Break the user query into sub-parts to find:
    1. A summary of the topic
    2. Suggested reference books
    3. Related articles or papers"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ])
    return {**state, "search_plan": response.content}


# === Node 2: Web Search ===
def search_node(state: AgentState) -> AgentState:
    results = serper_search(state["search_plan"])
    return {**state, "search_results": results}

# === Node 3: Summarize ===
def summarize_node(state: AgentState) -> AgentState:
    response = llm.invoke([
        SystemMessage(content="Summarize the following into: (1) Topic Summary, (2) Book References, (3) Article Links."),
        HumanMessage(content=state["search_results"])
    ])
    return {**state, "final_answer": response.content}


#LangGraph Flow
graph = StateGraph(AgentState)
graph.add_node("plan", plan_node)
graph.add_node("search", search_node)
graph.add_node("summarize", summarize_node)

graph.set_entry_point("plan")
graph.add_edge("plan", "search")
graph.add_edge("search", "summarize")
graph.add_edge("summarize", END)

runnable = graph.compile()

if __name__ == "__main__":
    user_input = input(" Enter a topic to research: ")
    result = runnable.invoke({"query": user_input})
    print("\n Research Output:\n")
    print(result["final_answer"])
