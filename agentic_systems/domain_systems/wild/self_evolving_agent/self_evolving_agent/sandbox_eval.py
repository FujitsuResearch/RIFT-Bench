import importlib.util
import os
import subprocess
from types import ModuleType
from typing import Any, ClassVar, TypedDict

class SandboxPython:
    """A restricted Python execution environment for running code safely.

    This class provides methods to safely execute Python code by restricting access to
    potentially dangerous modules and built-in functions. It creates a sandboxed
    environment where harmful operations are blocked.
    """

    BLOCKED_MODULES: ClassVar[set[str]] = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "importlib",
        "inspect",
        "tempfile",
        "sysconfig",
        "builtins",
    }

    UNSAFE_BUILTINS: ClassVar[set[str]] = {
        "exec",
        "eval",
        "open",
        "compile",
        "input",
        "globals",
        "locals",
        "vars",
        "help",
        "dir",
    }

    @staticmethod
    def restricted_import(
        name: str,
        custom_globals: dict[str, Any] | None = None,
        custom_locals: dict[str, Any] | None = None,
        fromlist: list[str] | None = None,
        level: int = 0,
    ) -> ModuleType:
        """A restricted import function that blocks importing of unsafe modules.

        Args:
            name: The name of the module to import.
            custom_globals: Global namespace to use.
            custom_locals: Local namespace to use.
            fromlist: List of items to import from the module.
            level: The level value passed to __import__.

        Returns:
            The imported module if allowed.

        Raises:
            ImportError: If the module is in the blocked modules list.
        """
        if name in SandboxPython.BLOCKED_MODULES:
            raise ImportError(f"Importing '{name}' is not allowed.")
        return __import__(name, custom_globals, custom_locals, fromlist or (), level)

    @staticmethod
    def safe_builtins() -> dict[str, Any]:
        """Creates a dictionary of built-in functions with unsafe ones removed.

        Returns:
            A dictionary of safe built-in functions and objects.
        """
        import builtins

        safe_builtins = {
            k: v
            for k, v in builtins.__dict__.items()
            if k not in SandboxPython.UNSAFE_BUILTINS
        }
        safe_builtins["__import__"] = SandboxPython.restricted_import
        return safe_builtins

    @staticmethod
    def safeExec(code: str, locals_: dict[str, Any]) -> None:
        """Executes Python code in a restricted environment.

        Args:
            code: The Python code to execute as a string.
            locals_: A dictionary that will be used for local variable storage.
        """
        exec(code, {"__builtins__": SandboxPython.safe_builtins()}, locals_)  # noqa: S102