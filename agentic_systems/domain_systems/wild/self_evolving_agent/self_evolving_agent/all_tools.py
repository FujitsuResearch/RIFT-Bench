from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, RetryPolicy
from dotenv import load_dotenv
import os
from langgraph.func import entrypoint, task
from langchain_openai import AzureChatOpenAI
import ast
from openai import OpenAI
from parse_and_separate import separate_definitions_and_usage, get_top_level_function_names
from tool_formatter import insert_docstring
from sandbox_eval import SandboxPython


# Setup the directory and filenames to store the tools generated.
# Useful for human debugging
ROOT = os.path.dirname(__file__)
TOOLS_DIR = os.path.join(ROOT, "tools")
MANIFEST_PATH = os.path.join(TOOLS_DIR, "manifest.json")


# Safety configuration: what is allowed and what is banned in the generated code
BANNED_IMPORTS = {"socket", "requests", "http", "ftplib", "paramiko", "subprocess", "multiprocessing", "sys", "os","shutil","importlib","inspect","tempfile","sysconfig","builtins"}
BANNED_NAMES = {"exec", "eval", "compile", "open", "input", "globals","locals","vars","help","dir","os.system", "popen", "run", "spawn"}
ALLOWED_MODULES = {"math", "json", "re", "datetime", "decimal", "numpy"}

def code_generation_and_execution(userQuery: str) -> str:
    """A tool for generating Python code based on the given code specification.
    This tool provides functionality to generate safe Python code according to the required specification"""
    client = OpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        base_url=os.getenv("AZURE_OPENAI_BASE_URL"),    
    )

    def generate_code(inputQuery: str) -> str:
        print("\n\ngenerate code")
        fixed_str = f""" Wrap code and example usage in one single python block including libraries to be imported.
        Only return code. DONT ADD ```python and ``` strings to the code"""
        # input = state["spec"], state["name"]
        response = client.responses.create(
            model=os.getenv("AZURE_DEPLOYMENT_NAME_CODEX"),
            # input="Write a Python function to perform a quicksort algorithm with comments."+fixed_str,
            input=inputQuery+fixed_str,
            reasoning={
                "effort": "high" # Use "low", "medium", or "high" for different reasoning depth
            },        
            # max_completion_tokens=1000,
        )
        # print(response)
        # print("\n\n")
        code=response.output_text
        return code

    
    def static_check(code: str) -> int:
        
        # print(f"STATIC_CHECK_CODE = \n\n{code}\n\n")
        if code.startswith("```python"):            
            code=code.removeprefix("```python")
            code=code.removesuffix("```")
        if code.startswith("```"):
            code=code.removeprefix("```")
            code=code.removesuffix("```")
        # print(f"STATIC_CHECK_CODE = \n\n{code}\n\n")
        try:
            tree = ast.parse(code)
        except Exception as e: 
            print(f"parse exception - error = {e} = \n{code}\n\n")           
            return -1

        for node in ast.walk(tree):
            # imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split(".")[0]
                    if mod in BANNED_IMPORTS:    
                        print("banned imports\n\n")                    
                        return -1
                    if mod not in ALLOWED_MODULES and not mod.startswith("numpy") and not mod.startswith("pandas"):
                        # conservative -- warn, but don't fail
                        pass
            if isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                if mod in BANNED_IMPORTS:   
                    print("banned imports2\n\n")                 
                    return -1
            # function calls / names
            if isinstance(node, ast.Call):
                # check if called function name is suspicious
                try:
                    if isinstance(node.func, ast.Attribute):
                        full = f"{ast.unparse(node.func)}"
                    else:
                        full = node.func.id
                except Exception:
                    print("not node or func\n\n")
                    full = None
                if full:
                    for banned in BANNED_NAMES:
                        if banned in full:
                            print("banned names\n\n")
                            return -1            
        # naive text search for obvious issues
        lowered = code.lower()
        for s in ("import os", "os.system", "subprocess", "socket", "requests", "eval("):
            if s in lowered:
                print("banned packages and keywords\n\n")
                return -1
        return 0

    def sandbox_execution(code: str) -> [str,int]:
        exec_locals: dict[str, Any] = {}

        try:
            SandboxPython.safeExec(code=code, locals_= exec_locals)
            res=exec_locals.get("result", "No result variable found.")
            return res, 0
        except Exception as e:
            print(f"An error occurred: {e!s}")
            return f"An error occurred: {e!s}",-1
        

    def check_code_format_and_save_as_tool(code: str)-> [str,str,str]:            

        print("\n\n\ncheck_code_format_and_save_as_tool invoked!!!!!")
        defs, usage, imports = separate_definitions_and_usage(code)
        function_names = get_top_level_function_names(code)

        if not defs and not function_names:
            return None, None, usage

        load_dotenv(override=True)
        util_llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
            model=os.getenv("AZURE_MODEL_NAME"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
            temperature=0.0,
        )

        importCode = "\n".join(imports)
        # print("import code = ",importCode)
        # re-usable code, hence save as tool. assuming only one function is returned by LLM
        decoratorStr = list()
        for x,y in zip(function_names,defs):
            if y.startswith("def"):
                # print("inside decoration loop")
                # a function definition
                prompt_str = f"""Please provide a one-line description of the following code: {y}. 
                Please include what input does the function take. Do not include function name in description."""
        
                response = util_llm.invoke(prompt_str)    
                # print("util_llm response: ",response.content)               
                
                code_description = response.content
                # Insert docstring after function definition line
                code_with_docstring = insert_docstring(y,x,code_description)
                # Now concatenate (prepend) CrewAI tool decorator, code_with_docstring, (append) tool handle and tool list        
                decoratorStr.append(f"""\n\n{code_with_docstring}""")

        
        # join all tool definitions code
        tool_defn_code = "\n\n".join(decoratorStr)
        full_tool_defn_code = importCode+"\n"+tool_defn_code
     
        return tool_defn_code, full_tool_defn_code, usage
            

    code = generate_code(userQuery)    
    error_static = static_check(code)
    res_str, error_sandbox = sandbox_execution(code)
    max_retry = 2
    num_retries=0
    # while ((error_static < 0) or (error_sandbox < 0)) and (num_retries < max_retry):
    while (error_static < 0) and (num_retries < max_retry):
        code = generate_code(userQuery)
        error_static = static_check(code)
        res_str, error_sandbox = sandbox_execution(code)        
        num_retries += 1

    # Whatever code is generated, check the format, separate usage part and save the import and tool definition
    only_fn_code, full_formatted_code, usage_code = check_code_format_and_save_as_tool(code)

    if only_fn_code and full_formatted_code:
        full_formatted_code = full_formatted_code.strip()
        print("formatted code: \n",full_formatted_code)
        filename = "scratchpad.py"
        with open(filename,"a") as f:
            with_new_line = "\n\n"+full_formatted_code
            f.write(with_new_line)

        return f"The code is:\n{full_formatted_code}\n and the execution result is stored."
    else:
        print("formatted code: \n",code)
        return f"The code is:\n{code} \nand the execution result is stored."
    
    return f"result={res_str}"


    