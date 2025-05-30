import os
import json
import httpx
import time
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage
from litellm import completion

load_dotenv()

memory = ConversationBufferMemory(return_messages=True)

# LLM wrapper
def llm_invoke(messages: list) -> str:
    formatted = [{"role": "system", "content": ""}]
    for msg in messages:
        if isinstance(msg, SystemMessage):
            formatted.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            formatted.append({"role": "user", "content": msg.content})
        else:
            formatted.append({"role": "assistant", "content": str(msg.content)})

    response = completion(
        model="gemini/gemini-2.0-flash",
        messages=formatted,
        api_key=os.getenv("GEMINI_API_KEY")
    )
    return response["choices"][0]["message"]["content"]


TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for up-to-date information, books, or articles related to the input query."
    },
    {
        "name": "fetch_url_content",
        "description": "Fetch and return the content of a URL, especially useful for extracting article or book details."
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient from sender email with body and subject."
    }
]

#Call MCP Tool over SSE 
def call_mcp_tool(tool_name: str, args: dict) -> str:
    print(f"\n[Tool Call] Tool: {tool_name}\nArguments: {args}\n", flush=True)
    mcp_url = os.getenv("MCP_URL")
    if not mcp_url:
        raise EnvironmentError("MCP_URL not set in environment variables.")

    payload = {"tool": tool_name, "args": args}

    with httpx.Client(timeout=None) as client:
        with client.stream("POST", mcp_url, json=payload, headers={"Accept": "text/event-stream"}) as response:
            if response.status_code != 200:
                raise RuntimeError(f"MCP SSE connection failed: {response.status_code} - {response.text}")

            collected = []
            for line in response.iter_lines():
                if line.startswith(b"event: tool_result"):
                    collected.append(line.decode().split("data: ", 1)[-1])
                elif line.startswith(b"event: done"):
                    break

            return "\n".join(collected).strip()

# Step 1: Planning
def plan_step(query: str) -> dict:
    past_messages = memory.load_memory_variables({})["history"]

    tool_info = "\n".join([f"- {t['name']}: {t['description']}" for t in TOOLS])
    system_prompt = f"""You are a research assistant. Given the user query, determine:
1. Whether tool use is required. If the question is factual and common (e.g., 'What is the capital of France?'), no tools are needed.
2. If tools are needed, specify:
   - The reasoning for using tools.
   - The most appropriate tool(s) from this list:
{tool_info}

Return ONLY valid JSON in this format (no explanation or prose, just the JSON object):

{{
  "use_tools": true or false,
  "tool_name": "tool_name_if_applicable",
  "tool_args": {{ relevant arguments }},
  "reasoning": "Why you chose this tool and these args"
}}"""

    messages = [SystemMessage(content=system_prompt), *past_messages, HumanMessage(content=query)]
    plan_response = llm_invoke(messages)

    try:
        json_str = plan_response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[-1].split("```")[0].strip()
        plan = json.loads(json_str)
    except Exception as e:
        print(f"[Plan Error] Failed to parse plan: {e}", flush=True)
        plan = {"use_tools": False, "reasoning": "Fallback: parsing failed."}

    memory.save_context({"input": query}, {"output": plan_response})
    print(f"\n[Plan Result]\n{json.dumps(plan, indent=2)}", flush=True)
    return plan


#Step 2: Executing the Tool
def execute_tool_step(plan: dict) -> str:
    if isinstance(plan.get("use_tools"), str):
        use_tools = plan["use_tools"].lower() == "true"
    else:
        use_tools = plan.get("use_tools", False)

    if not use_tools:
        return plan.get("reasoning", "Tool use not required.")

    try:
        tool_name = plan["tool_name"]
        args = plan.get("tool_args", {})
        result = call_mcp_tool(tool_name, args)
        return json.loads(result) if result.startswith("[") else result
    except Exception as e:
        return f"[Tool Error] {e}"

#Step 3: Summarize the response
def summarize_step(query: str, tool_result: str | None) -> str:
    past_messages = memory.load_memory_variables({})["history"]

    system_prompt = """You are a helpful research assistant. Your task is to:
1. Summarize the given topic.
2. Recommend 2–3 relevant books (title and author).
3. List 2–3 relevant online articles (title + URL).

If tool_result is provided, use it as context for summarizing the topic.

Respond in the following Markdown format:

**Summary:** <summary here>

**Recommended Books:**
- *Title* by Author
- *Title* by Author

**Relevant Articles:**
- [Title](URL) - Optional short description
"""

    messages = [SystemMessage(content=system_prompt), *past_messages]

    if tool_result:
        messages.append(HumanMessage(content=f"Topic: {query}\n\nHere is some content from a tool:\n{tool_result}"))
    else:
        messages.append(HumanMessage(content=f"Topic: {query}"))

    final_output = llm_invoke(messages)
    memory.save_context({"input": query}, {"output": final_output})

    #print("\n 📑 Final Output:\n", final_output, flush=True)
    return final_output

#Full Flow 
def run_agent(query: str):
    print("\n[Step 1] Planning...\n")
    plan = plan_step(query)

    print("\n[Step 2] Tool Execution (if needed)...\n")
    tool_result = execute_tool_step(plan)

    print("\n[Step 3] Summarization...\n")
    summary = summarize_step(query, tool_result)

    return summary

if __name__ == "__main__":
    user_input = input("📖 Enter the topic name: ")
    final_output = run_agent(user_input)
    print("\📑 Final Output: Your rsearch is ready\n")
    print(final_output.strip(), flush=True)
