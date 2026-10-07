from crewai_tools import CodeInterpreterTool
from typing import List


class InterpreterTool(CodeInterpreterTool):
    def __init__(self):
        super().__init__()

    def run_code_safety(self, code: str, libraries_used: List[str]) -> str:
        return super().run_code_in_restricted_sandbox(code)
