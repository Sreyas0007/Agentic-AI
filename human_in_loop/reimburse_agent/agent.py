import os 
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool, LongRunningFunctionTool
from typing import Any
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
request = "request"

# 1. Define the long running function
def ask_for_approval(purpose: str, amount: float) -> dict[str, Any]:
    """Ask for approval for the reimbursement.
    create a ticket for the approval.
    Send a notification to the approver with the link of the ticket. """

    return {'status': 'pending', 'approver': 'Sean Zhou', 'purpose' : purpose, 'amount': amount, 'ticket-id': 'approval-ticket-1'}

def reimburse(purpose: str, amount: float) -> str:
    """Reimburse the amount of money to the employee.
    send the reimbrusement request to payment vendor """
    return {'status': 'ok'}

# 2. Wrap the function with LongRunningFunctionTool
long_running_tool = LongRunningFunctionTool(func=ask_for_approval)

request_agent = LlmAgent(
    name = "approval_request_agent",
    model = "gemini-1.5-flash",
    description = "Agent that prepares a request for reimbursement approval",
    instruction = """Your a reimbursement approval request prepare agent. When user gives amount and purpose
    as inputs, prepare a request on the user inputs and send the request to the vendor(human) for approval. 
    Without the approval of vendor do not proceed. Take the response of vendor and then proceed.
    use the ask_for_approval tool.

    Important note:
    1) send the output only after getting approver response, without the response don't give any output.
    2) Do not generate the human(Approver) response or human approval. Clearly ask the approver for approver response 
    as a part of human response.
    3) After you get the human response only, you pass the user request and approver response to the next agent.
    """,
    tools = [LongRunningFunctionTool(func = ask_for_approval)],
    output_key = request

)

reimbursement_approval_agent = LlmAgent(
    name = "approval_agent",
    model = "gemini-1.5-flash",
    description = "Agent that process the request and approves the reimbursement request",
    instruction = f"""Your a reimbursement response agent. Based on the {{request}} sent by the request agent,
    send the approver response back to the user. You are sending a human(approver) response,
    like aprroved or rejected status of reimbursement  request is sent to the user as output.

    
    Your input:
    {{request}} which has user request details and approver response

    Your output:
    approver response to the user

    use the reimburse tool. 
    """,
    tools = [LongRunningFunctionTool(func = reimburse)],
    output_key= "output"
)

pipe_agent = SequentialAgent(
    name = "reimbursement_agent",
    description= "runs sub agents in a sequence",
    sub_agents= [request_agent, reimbursement_approval_agent]
)

root_agent = pipe_agent
