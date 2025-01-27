from io import StringIO
import os
import random
import re
import sys
import traceback
from typing import Callable, Tuple
from dataclasses import dataclass, field
from git import Repo
from pathlib import Path
import requests

from dotenv import load_dotenv
load_dotenv()


from haystack.dataclasses import ChatMessage, ChatRole
from haystack.tools import create_tool_from_function
from haystack.components.tools import ToolInvoker
from haystack.components.generators.chat import OpenAIChatGenerator

HANDOFF_TEMPLATE = "Transferred to: {agent_name}. Adopt persona immediately."
HANDOFF_PATTERN = r"Transferred to: (.*?)(?:\.|$)"

PATH_TO_EX_DIR = "./src/haystack/examples/"


@dataclass
class SwarmAgent:
    name: str = "SwarmAgent"
    llm: object = OpenAIChatGenerator(model="gpt-4o")
    instructions: str = "You are a helpful Agent"
    functions: list[Callable] = field(default_factory=list)

    def __post_init__(self):
        self._system_message = ChatMessage.from_system(self.instructions)
        self.tools = [create_tool_from_function(fun) for fun in self.functions] if self.functions else None
        self._tool_invoker = ToolInvoker(tools=self.tools, raise_on_failure=False) if self.tools else None

    def run(self, messages: list[ChatMessage]) -> Tuple[str, list[ChatMessage]]:
        # generate response
        agent_message = self.llm.run(messages=[self._system_message] + messages, tools=self.tools)["replies"][0]
        new_messages = [agent_message]

        if agent_message.text:
            print(f"\n{self.name}: {agent_message.text}")

        if not agent_message.tool_calls:
            return self.name, new_messages

        # handle tool calls
        for tc in agent_message.tool_calls:
            # trick: Ollama do not produce IDs, but OpenAI and Anthropic require them.
            if tc.id is None:
                tc.id = str(random.randint(0, 1000000))
        tool_results = self._tool_invoker.run(messages=[agent_message])["tool_messages"]
        new_messages.extend(tool_results)

        # handoff
        last_result = tool_results[-1].tool_call_result.result
        match = re.search(HANDOFF_PATTERN, last_result)
        new_agent_name = match.group(1) if match else self.name

        return new_agent_name, new_messages
    
# tools
    
def handoff_to_debugger():
    """Pass to this agent for debugging created code."""
    return HANDOFF_TEMPLATE.format(agent_name="Debugging Agent")

def handoff_to_tester():
    """Pass to this agent for testing created code."""
    return HANDOFF_TEMPLATE.format(agent_name="Testing Agent")

def handoff_to_coder():
    """Pass to this agent for modifying existing code."""
    return HANDOFF_TEMPLATE.format(agent_name="Coding Agent")

def handoff_to_github():
    """Pass to this agent for interactions with github."""
    return HANDOFF_TEMPLATE.format(agent_name="Github Agent")

def cloneGithubRepo(repo_url: str, dir_name: str) -> str:
    """
    A tool cloning a repository from GitHub for local access to the relative path `./src/haystack/examples/`
    into the directory with the specified dir_name.
    """

    path = PATH_TO_EX_DIR + dir_name
    Path(path).mkdir(parents=True, exist_ok=True)
   
    Repo.clone_from(repo_url, path)
    return path

def retrieveGithubIssue(repo_url:str, issue_number:int) -> str:
    """
    Fetch the title and description of a GitHub issue.

    Parameters:
        repo_url (str): The URL of the GitHub repository (e.g., 'https://github.com/owner/repo').
        issue_number (int): The issue number to fetch details for.

    Returns:
        str: A string containing the issue title and description, or an error message.
    """
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

def executeCode(path_to_module:str, **kwargs:dict) -> str:
    """
    A tool executing python code. If execution fails, traceback is returned. Otherwise, stdout output is returned.
    """

    # Check if filename is an absolute path
    if os.path.isabs(path_to_module) and PATH_TO_EX_DIR in path_to_module:
        path = path_to_module
    else:
        # Use base directory for relative paths
        base_dir = PATH_TO_EX_DIR
        path = os.path.join(base_dir, path_to_module)

    try:
        with open(path) as file:
            with Capturing() as output:
                exec(file.read(), kwargs)
    except:
        return(traceback.format_exc())
    return output
		
def saveCode(code:str, filename:str) -> str:
    """
    A tool saving code locally.
    """
    
    # Ensure filename has the correct extension
    if not filename.endswith(".py"):
        filename += ".py"

    # Check if filename is an absolute path
    if os.path.isabs(filename) and PATH_TO_EX_DIR in filename:
        path = filename
    else:
        # Use base directory for relative paths
        base_dir = PATH_TO_EX_DIR
        path = os.path.join(base_dir, filename)

    with open(path, "w") as file:
        file.write(code)
    return path
	
def readCode(path_to_code:str) -> str:
    """
    A tool reading in code from a file.
    """
    # Ensure the file path is absolute
    if not os.path.isabs(path_to_code):
        base_dir = PATH_TO_EX_DIR
        path_to_code = os.path.join(base_dir, path_to_code)

    # Check if the file exists
    if not os.path.exists(path_to_code):
        raise FileNotFoundError(f"File not found: {path_to_code}")

    # Read and return the file content
    with open(path_to_code, "r") as file:
        code = file.readlines()
    return code

def findFile(filename:str):
    """
    Search for a file within a directory and its sub-directories.

    Parameters:
        filename (str): The name of the file to search for.

    Returns:
        str: The absolute path to the file if found, or an error message if not found.
    """
    try:
        for root, _, files in os.walk(PATH_TO_EX_DIR):
            if filename in files:
                return os.path.abspath(os.path.join(root, filename))
        return f"Error: File '{filename}' not found in directory ./src/haystack/examples."
    except Exception as e:
        return f"Error: An error occurred while searching for the file: {str(e)}"

    

# agents

coding_agent = SwarmAgent(
    name = "Coding Agent",
    instructions=("You are the Coding Agent. The other agents are Debugging Agent, Testing Agent and Github Agent. You are an experienced programmer."
                "You can write professional python code for any specified coding problem."
                "You can extend and modify existing code."
                "You can save generated code in a file with the tool saveCode."
                "If you run into any bugs, do not fix them. Instead return a description of the bugs."
                "In this case, hand off to debugging agent by using the handoff_to_debugger tool."
                "If you are asked test the code, hand off to testing agent by using handoff_to_tester tool."
                "If interactions with github are necessary, hand off to github agent."
                "If you have issues finding a specific file, make use of the findFile tool."
                "Execute the task step by step and explain your steps while you are executing them, also mention any used tools."),
    functions=[saveCode, readCode, findFile, handoff_to_debugger, handoff_to_tester, handoff_to_github],
)

testing_agent = SwarmAgent(
    name = "Testing Agent",
    instructions=("You are the Testing Agent."
                "The other agents are Coding and Debugging Agent."
                "You are a software engineer with expertise in software testing."
                "Your task is to write thorough tests in Python for the provided code.\
                        Follow these steps: \
                        1. Read the provided code. \
                        2. Write test cases to cover edge cases and typical usage. \
                        3. Save the test code to a file using the saveCode tool. \
                        4. Execute the test file using the `execution_tool` and report the results. "
                "If you have issues finding a specific file, make use of the findFile tool."
                "If the tests ran successfully, report back that testing was a success."
                "If the tests do not run successfully, report back issues with the code and hand off to coder."
                "If interactions with github are necessary, hand off to github agent."),
    functions=[saveCode, readCode, executeCode, findFile, handoff_to_coder, handoff_to_debugger, handoff_to_github]
)

debugging_agent = SwarmAgent(
    name = "Debugging Agent",
    instructions=("You are the Debugging Agent."
                "The other agents are Coding and Testing Agent."
                "You are an experienced programmer."
				"First, read in the existing code."
				"To find bugs in the code, you can execute it and make use of possible error messages."
                "Furthermore, you can use previous chat messages to find possible bugs."
                "Execute the task step by step and explain your steps while you are executing them, also mention any used tools."
				"After you finish debugging, report back if any bugs were found and if so, how they can be fixed."
                "If there are bugs to fix, hand off to coding agent for fixing the bugs by using the tool handoff_to_coder."
                "If interactions with github are necessary, hand off to github agent."
                "If you have issues finding a specific file, make use of the findFile tool."),
    functions=[readCode, executeCode, findFile, handoff_to_tester, handoff_to_coder, handoff_to_github],
)

github_agent = SwarmAgent(
    name = "Github Agent",
    instructions = ("You are the Github Agent."
                    "The other Agents are Coding, Testing and Debugging Agent."
                    "You are an expert in interacting with Github, specifically cloning repositories and retrieving issues."
                    "If you need to clone a repository, use the cloneGithubRepo tool."
                    "If you need to retrieve an issue from a repository, use the retrieveGithubIssue tool."
                    "You are not responsible for coding, debugging or testing."
                    "For code generation tasks, hand off to coding agent."
                    "For debugging tasks, hand off to debugging agent."
                    "For testing tasks, hand off to testing agent."
                    "Explain any thoughts you have and any steps you take. Mention every tool you use."),
    functions=[cloneGithubRepo, retrieveGithubIssue, handoff_to_coder, handoff_to_debugger, handoff_to_tester],
)




def run(task_prompt: str):

    agents = {agent.name: agent for agent in [coding_agent, debugging_agent, testing_agent, github_agent]}

    messages = [ChatMessage.from_user(task_prompt)]
    current_agent_name = "Coding Agent"

    while True:
        agent = agents[current_agent_name]

        if messages[-1].role == ChatRole.ASSISTANT:
            break

        current_agent_name, new_messages = agent.run(messages)
        messages.extend(new_messages)


if __name__ == "__main__":

    user_input = input("User: ")
    run(user_input)