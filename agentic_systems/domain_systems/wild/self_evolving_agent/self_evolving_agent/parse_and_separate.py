import ast
from typing import List, Tuple


def separate_definitions_and_usage(code: str) -> Tuple[List[str], List[str]]:
    """
    Separates function/class definitions and example usage code
    from a Python source string.
    """
    tree = ast.parse(code)

    definitions = []
    usage = []
    imports = []
    for node in tree.body:
        source = ast.get_source_segment(code, node)

        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            definitions.append(source)
        elif (isinstance(node,ast.Import) or isinstance(node,ast.ImportFrom)):
            imports.append(source)
        elif (isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"):
            usage.append(source)
        else:
            usage.append(source)

    return definitions, usage, imports

def get_top_level_function_names(code: str) -> List[str]:
    tree = ast.parse(code)

    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]


# code = open("example.py").read()

# defs, usage = separate_definitions_and_usage(code)

# print("=== DEFINITIONS ===")
# # # for now assume that only one function is returned as code by codex. 
# # will require improvment later- to be extended for multiple functions or class!
# print("\n\n".join(defs))

# print("\n=== USAGE ===")
# print("\n".join(usage))