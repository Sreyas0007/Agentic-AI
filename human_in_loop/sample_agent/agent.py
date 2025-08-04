import os
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool

request = "request"
approval = "approval"

async def external_approval_tool( amount: float, reason: str):
    f"""
    Args:
    1. Take details (e.g., request_id, {amount}, {reason}).
    2. Send these details to a human review system (e.g., via API).
    3. Poll or wait for the human response (approved/rejected).
    4. Return the human's decision.    
    """

approval_tool = FunctionTool(func= external_approval_tool)

prepare_request = LlmAgent(
    name = "prepare_request_agent",
    model = "gemini-1.5-flash",
    description= "An agent that prepares a aprroval request",
    instruction = """ prepare a approval request deatils based on the user input.,
    store the amount and reason in a state, like set state['approval_amount'] and state['approval_reason'].

    """,
    output_key= request

)

request_approval = LlmAgent(
    name = "request_approval_agent",
    model = "gemini-2.0-flash",
    description = "An agent that takes human response to approve the request",
    instruction = f""" You are a request approval agent, you take the request details and ask the vendor for approval.
    You clearly should ask human(vendor) for approval and without human approval you cannot proceed.
    Your inputs are:
    {{request}}

    Pass the input {{request}} to the human and take the human approval.

    Your output is:
    Human response

    use the external_approval_tool, amount from state['amount'] 
    and reason from state['reason'] for request approval
    """,
    tools = [approval_tool],
    output_key = approval

)

decision_processing = LlmAgent(
    name = "decision_process_agent",
    model = "gemini-1.5-flash",
    instruction = """ You are an agent that process the human decision. 
    If the human_response is aprroved then inform user that request is approved, 
    if the human_response is rejected then inform the user that request is rejected.
    """
)

root_agent = SequentialAgent(
    name = "workflow_agent",
    description= "Runs the sub-agents in an ordered sequence",
    sub_agents = [prepare_request, request_approval, decision_processing]
)
