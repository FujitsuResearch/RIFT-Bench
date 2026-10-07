
import os
import json
import ast
from typing import  Literal, List, Tuple, Type



def insert_docstring(
    source_code: str,
    function_name: str,
    docstring: str
) -> str:
    """
    Insert or replace a docstring at the start of a function.
    Returns updated source code.
    """

    tree = ast.parse(source_code)

    class DocstringInserter(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef):
            if node.name == function_name:
                doc_node = ast.Expr(value=ast.Constant(value=docstring))

                # Replace existing docstring
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    node.body[0] = doc_node
                else:
                    node.body.insert(0, doc_node)

            return node

        visit_AsyncFunctionDef = visit_FunctionDef

    tree = DocstringInserter().visit(tree)
    ast.fix_missing_locations(tree)

    return ast.unparse(tree)          
                