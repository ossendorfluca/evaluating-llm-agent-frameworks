# based on this example: https://github.com/run-llama/multi-agent-concierge

import asyncio
from io import StringIO
import os
from pathlib import Path
import sys
import traceback
from dotenv import load_dotenv
import requests
load_dotenv()

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import BaseTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

from src.llama_index.workflow import (
    AgentConfig,
    SoftwareEngineeringAgent,
    ProgressEvent,
)
from src.llama_index.utils import FunctionToolWithContext

from git import Repo

PATH_TO_EX_DIR = "./src/llama_index/examples/"


def get_initial_state() -> dict:
    return {
        "username": None,
        "session_token": None,
        "account_id": None,
        "account_balance": None,
    }

# Tools

def cloneGithubRepo(ctx: Context, repo_url: str, dir_name: str) -> str:
    """
    A tool cloning a repository from GitHub for local access to the relative path `./src/llama_index/examples/`
    into the directory with the specified dir_name.
    """
    ctx.write_event_to_stream(
        ProgressEvent(msg=f"Cloning GitHub Repo under URL: {repo_url} to directory {dir_name}.")
    )
    path = PATH_TO_EX_DIR + dir_name
    Path(path).mkdir(parents=True, exist_ok=True)
    Repo.clone_from(repo_url, path)
    return path

clone_tool = FunctionToolWithContext.from_defaults(fn=cloneGithubRepo)

def retrieveGithubIssue(ctx: Context, repo_url:str, issue_number:int) -> str:
    """
	A tool retrieving an issue from a Github Repository via the url to the issue.
	"""
    ctx.write_event_to_stream(
            ProgressEvent(msg=f"Retrieving the issue number {issue_number} from repository: {repo_url}.")
        )
    try:
        # Extract the owner and repo name from the URL
        url_parts = repo_url.rstrip('/').split('/')
        if len(url_parts) < 2:
            raise ValueError("Invalid repository URL format.")
        
        owner, repo = url_parts[-2], url_parts[-1]

        # Construct the API URL
        api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"

        # Make a GET request to fetch the issue details
        response = requests.get(api_url)

        if response.status_code == 200:
            issue_data = response.json()
            title = issue_data.get("title", "No title provided")
            description = issue_data.get("body", "No description provided")
            return f"Title: {title}\nDescription: {description}"
        else:
            return f"Error: Failed to fetch issue details. HTTP Status Code: {response.status_code}, Message: {response.json().get('message', 'Unknown error')}"
    except Exception as e:
        return f"Error: An error occurred: {str(e)}"

issue_tool = FunctionToolWithContext.from_defaults(fn=retrieveGithubIssue)

# https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def executeCode(ctx: Context, path_to_module:str, **kwargs:dict) -> str:
    """
    A tool executing python code. If execution fails, traceback is returned. Otherwise, stdout output is returned.
    """

    if PATH_TO_EX_DIR in path_to_module:
        path = path_to_module
    else:
        # Use base directory for relative paths
        base_dir = PATH_TO_EX_DIR
        path = os.path.join(base_dir, path_to_module)

    ctx.write_event_to_stream(
            ProgressEvent(msg=f"Executing code with path: {path}.")
        )

    try:
        with open(path) as file:
            with Capturing() as output:
                exec(file.read(), kwargs)
    except:
        return(traceback.format_exc())
    return output
		
execution_tool = FunctionToolWithContext.from_defaults(fn=executeCode)

def saveCode(ctx: Context, code:str, filename:str) -> str:
    """
    A tool saving code locally.
    """
    
    # Ensure filename has the correct extension
    if not filename.endswith(".py"):
        filename += ".py"

    if PATH_TO_EX_DIR in filename:
        path = filename
    else:
        # Use base directory for relative paths
        base_dir = PATH_TO_EX_DIR
        path = os.path.join(base_dir, filename)

    ctx.write_event_to_stream(
        ProgressEvent(msg=f"Saving code at path: {path}.")
    )

    with open(path, "w") as file:
        file.write(code)
    return path
	
saving_tool = FunctionToolWithContext.from_defaults(fn=saveCode)

def readCode(ctx: Context, path_to_code:str) -> str:
    """
    A tool reading in code from a file.
    """
    if PATH_TO_EX_DIR in path_to_code:
        path = path_to_code
    else:
        # Use base directory for relative paths
        base_dir = PATH_TO_EX_DIR
        path = os.path.join(base_dir, path_to_code)

    # Check if the file exists
    if not os.path.exists(path_to_code):
        raise FileNotFoundError(f"File not found: {path_to_code}")

    ctx.write_event_to_stream(
        ProgressEvent(msg=f"Reading in code at path: {path_to_code}.")
    )

    # Read and return the file content
    with open(path_to_code, "r") as file:
        code = file.readlines()
    return code

reading_tool = FunctionToolWithContext.from_defaults(fn=readCode)

def findFile(ctx:Context, filename:str) -> str:
    """
    Search for a file within a directory and its sub-directories.
    """
    ctx.write_event_to_stream(
        ProgressEvent(msg=f"Looking for file: {filename}.")
    )
    try:
        for root, _, files in os.walk(PATH_TO_EX_DIR):
            if filename in files:
                return os.path.abspath(os.path.join(root, filename))
        return f"Error: File '{filename}' not found in directory ./src/llama_index/examples."
    except Exception as e:
        return f"Error: An error occurred while searching for the file: {str(e)}"

finding_tool = FunctionToolWithContext.from_defaults(fn=findFile)



def get_agent_configs() -> list[AgentConfig]:
    return [
        AgentConfig(
            name="Coding Agent",
            description="Writes python code",
            system_prompt="You are the Coding Agent. The other agents are Debugging and Testing Agent. You are an experienced programmer. \
                    You can write professional python code for any specified coding problem. \
                    You can extend and modify existing code. \
                    You can save generated code in a file with the tool saving_tool. \
                    You can execute the code with sample input by using the tool execution_tool. \
                    If you run into any bugs, do not fix them. Instead return a description of the bugs. \
                    If you are asked test the code, request a transfer to another agent by using RequestTransfer tool. \
                    Execute the task step by step and explain your steps while you are executing them, also mention any used tools. \
                    Ignore any transfer statements that should transfer to yourself. If it is a transfer to another agent, use the RequestTransfer tool.",
            tools=[saving_tool, reading_tool, execution_tool, finding_tool],
        ),
        AgentConfig(
            name="Debugging Agent",
            description="Debugs python code",
            system_prompt="You are the Debugging Agent. The other agents are Coding and Testing Agent. You are an experienced programmer. \
					First, read in the existing code. \
					To find bugs in the code, you can execute it and make use of possible error messages. \
					If you cannot think of a fix for the bug, you can use the websearch tool to find answers online. \
					After you finish debugging, save the modified code. \
					Execute the task step by step and explain your steps while you are executing them, also mention any used tools. \
                    Ignore any transfer statements that should transfer to yourself. If it is a transfer to another agent, follow the instructions.",
            tools=[saving_tool, reading_tool, execution_tool, finding_tool],
        ),
        AgentConfig(
            name="Testing Agent",
            description="Tests python code",
            system_prompt="""You are the Testing Agent. The other agents are Debugging and Coding Agent. You are a software engineer with expertise in software testing.
                        Your task is to write thorough tests in Python for the provided code. 
                        Follow these steps: 
                        1. Read the provided code. 
                        2. Write test cases to cover edge cases and typical usage. 
                        3. Save the test code to a file using the `saving_tool`. 
                        4. Execute the test file using the `execution_tool` and report the results. 
                        If the tests run successfully or there are failures, return a final report. 
                        Ignore any transfer statements that should transfer to yourself. If it is a transfer to another agent, follow the instructions.
                        """,
            tools=[saving_tool, reading_tool, execution_tool, finding_tool],
        ),
        AgentConfig(
            name="Github Agent",
            description="Handles interactions with github.",
            system_prompt="""You are the Github Agent.
                    The other Agents are Coding, Testing and Debugging Agent.
                    You are an expert in interacting with Github, specifically cloning repositories and retrieving issues.
                    If you need to clone a repository, use the cloneGithubRepo tool.
                    If you need to retrieve an issue from a repository, use the retrieveGithubIssue tool.
                    You are not responsible for coding, debugging or testing.
                    For code generation tasks, hand off to coding agent.
                    For debugging tasks, hand off to debugging agent.
                    For testing tasks, hand off to testing agent.
                    Explain any thoughts you have and any steps you take. Mention every tool you use.
                    Ignore any transfer statements that should transfer to yourself. If it is a transfer to another agent, follow the instructions.""",
        tools=[issue_tool, clone_tool],
        ),
    ]

# main function for interactive chat
"""
async def main():

    llm = OpenAI(model="gpt-4o", temperature=0.4)
    memory = ChatMemoryBuffer.from_defaults(llm=llm)
    initial_state = get_initial_state()
    agent_configs = get_agent_configs()
    workflow = SoftwareEngineeringAgent(timeout=None)


    handler = workflow.run(
        user_msg="Hello!",
        agent_configs=agent_configs,
        llm=llm,
        chat_history=[],
        initial_state=initial_state,
    )

    while True:
        async for event in handler.stream_events():
            if isinstance(event, ProgressEvent):
                print(f"SYSTEM >> {event.msg}")

        result = await handler
        print(f"AGENT >> {result['response']}")

        # update the memory with only the new chat history
        for i, msg in enumerate(result["chat_history"]):
            if i >= len(memory.get()):
                memory.put(msg)

        user_msg = input("USER >> ")
        if user_msg.strip().lower() in ["exit", "quit", "bye"]:
            break

        # pass in the existing context and continue the conversation
        handler = workflow.run(
            ctx=handler.ctx,
            user_msg=user_msg,
            agent_configs=agent_configs,
            llm=llm,
            chat_history=memory.get(),
            initial_state=initial_state,
        )
"""

# run function for single response chat

async def run(user_input):
    llm = OpenAI(model="gpt-4o", temperature=0.4)
    memory = ChatMemoryBuffer.from_defaults(llm=llm)
    initial_state = get_initial_state()
    agent_configs = get_agent_configs()
    workflow = SoftwareEngineeringAgent(timeout=None)

    handler = workflow.run(
        user_msg=user_input,
        agent_configs=agent_configs,
        llm=llm,
        chat_history=[],
        initial_state=initial_state,
    )


    async for event in handler.stream_events():
        if isinstance(event, ProgressEvent):
            print(f"SYSTEM >> {event.msg}")

        result = await handler
        print(f"AGENT >> {result['response']}")

        # update the memory with only the new chat history
        for i, msg in enumerate(result["chat_history"]):
            if i >= len(memory.get()):
                memory.put(msg)


if __name__ == "__main__":
    user_input = input("User: ")
    asyncio.run(run(user_input))