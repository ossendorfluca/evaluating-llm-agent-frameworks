from io import StringIO
import os
from pathlib import Path
import sys
import traceback
from git import Repo
import requests
from transformers.agents import ManagedAgent, ReactCodeAgent, PythonInterpreterTool
from transformers.agents.llm_engine import MessageRole, get_clean_message_list
from transformers import Tool

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

PATH_TO_EX_DIR = "./src/hugging_face/examples/"

# model setup 

openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}

class OpenAIEngine:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def __call__(self, messages, stop_sequences=[]):
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
            temperature=0.5,
        )
        return response.choices[0].message.content

llm_engine = OpenAIEngine()

# tools setup

class SaveCode(Tool):
    name = "saving_tool"
    description = """
    This is a tool that saves python code to a file. It returns the path to the file."""

    inputs = {
        "code": {
            "type": "string",
            "description": "the python code to be saved to a file",
        },
        "filename": {
            "type": "string",
            "description": "the name of the file, without the path, but with .py ending",
        }
    }
    output_type = "string"

    def forward(self, code: str, filename:str):

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
    
class ReadCode(Tool):
    name = "reading_tool"
    description = """
    This is a tool that reads in python code from a file."""

    inputs = {
        "path_to_code": {
            "type": "string",
            "description": "the relative path to the code including src/hugging_face/examples/ and the filename",
        }
    }
    output_type = "string"

    def forward(self, path_to_code: str):

        if PATH_TO_EX_DIR not in path_to_code:
            path_to_code = PATH_TO_EX_DIR + path_to_code
        # Check if the file exists
        if not os.path.exists(path_to_code):
            raise FileNotFoundError(f"File not found: {path_to_code}")
        

        # Read and return the file content
        with open(path_to_code, "r") as file:
            code = file.readlines()
        return code


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
        
class ExecuteCode(Tool):
    name = "execution_tool"
    description = """
    This is a tool that executes python code. It returns the captured output."""

    inputs = {
        "path_to_module": {
            "type": "string",
            "description": "path to the module that should be executed",
        },
        "**kwargs": {
            "type": "string",
            "description": "optional parameters to be passed on to the function that is executed",
        }
    }
    output_type = "string"

    def forward(self, path_to_module: str, **kwargs:dict):

        if PATH_TO_EX_DIR in path_to_module:
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
    
class FindFile(Tool):
    name = "finding_tool"
    description = """
    This is a tool that finds a file in the ./src/hugging_face/examples directory. It returns the absolute path to the file."""

    inputs = {
        "filename": {
            "type": "string",
            "description": "name of the file to search for",
        },
    }
    output_type = "string"

    def forward(self, filename: str):
        try:
            for root, _, files in os.walk(PATH_TO_EX_DIR):
                if filename in files:
                    return os.path.abspath(os.path.join(root, filename))
            return f"Error: File '{filename}' not found in directory ./src/hugging_face/examples."
        except Exception as e:
            return f"Error: An error occurred while searching for the file: {str(e)}"

class RetrieveIssue(Tool):
    name = "issue_tool"
    description = """
    This is a tool that retrieves an issue from a github repository. It returns the title and description of the issue."""

    inputs = {
        "repo_url": {
            "type": "string",
            "description": "url to the github repository",
        },
        "issue_number": {
            "type": "integer",
            "description": "number of the github issue",
        },
    }
    output_type = "string"

    def forward(self, repo_url: str, issue_number:int):
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
        
class CloneRepository(Tool):
    name = "cloning_tool"
    description = """
    This is a tool that clones a github repository. It returns the path to the cloned repository."""

    inputs = {
        "repo_url": {
            "type": "string",
            "description": "url to the github repository",
        },
        "dir_name": {
            "type": "string",
            "description": "directory name to clone the repository into"
        }
    }
    output_type = "string"

    def forward(self, repo_url: str, dir_name:str):
   
        path = PATH_TO_EX_DIR + dir_name
        Path(path).mkdir(parents=True, exist_ok=True)
    
        Repo.clone_from(repo_url, path)
        return path
        

# agents setup

coding_agent = ReactCodeAgent(tools=[ReadCode(), SaveCode(), FindFile()], llm_engine=llm_engine, additional_authorized_imports=[])

managed_coding_agent = ManagedAgent(
    agent=coding_agent,
    name="coding_task",
    description="Writes python code. Give it its task as an argument."
)

testing_agent = ReactCodeAgent(tools=[ExecuteCode(), ReadCode(), SaveCode(), FindFile()], llm_engine=llm_engine, additional_authorized_imports=['unittest', 'sys'])

managed_testing_agent = ManagedAgent(
    agent=testing_agent,
    name="testing_task",
    description="Tests python code. Give it its task as an argument."
)

debugging_agent = ReactCodeAgent(tools=[ExecuteCode(), ReadCode(), FindFile()], llm_engine=llm_engine)

managed_debugging_agent = ManagedAgent(
    agent=debugging_agent,
    name="debugging_task",
    description="Debugs python code. Give it its task as an argument."
)

github_agent = ReactCodeAgent(tools=[RetrieveIssue(), CloneRepository()], llm_engine=llm_engine)

managed_github_agent = ManagedAgent(
    agent=github_agent,
    name="github_task",
    description="Clones github repositories and retrieves github issues. Give it its task as an argument."
)

manager_agent = ReactCodeAgent(
    tools=[PythonInterpreterTool()], 
    llm_engine=llm_engine, 
    managed_agents=[managed_coding_agent, managed_debugging_agent, managed_testing_agent, managed_github_agent],
    additional_authorized_imports=[]
)

# main run function
def run(task_prompt:str):
    try:
        manager_agent.run("Make use of the four given agents to solve the tasks given in the user input. Do not solve any tasks yourself. -- User Input: " + task_prompt)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    usr_input = input("User: ")
    run(usr_input)