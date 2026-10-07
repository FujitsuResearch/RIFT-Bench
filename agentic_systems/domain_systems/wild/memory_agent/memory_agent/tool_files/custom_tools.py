from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from tool_files.image_gen_tool import ImageGenTool
from langchain_core.tools import Tool, StructuredTool
import wikipedia
from playwright.sync_api import sync_playwright
import os
import random
from PyPDF2 import PdfReader
import numpy as np
import numexpr as ne
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from tool_files.sql_tool import Database
import sys
import hashlib
from typing import Optional, List, Union
import asyncio


def setup_paths():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_directory)
    grandparent_dir = os.path.dirname(parent_dir)

    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    if grandparent_dir not in sys.path:
        sys.path.append(grandparent_dir)


setup_paths()
load_dotenv(override=True)

# === SQL TOOLS ===
db = Database(
    db_url="https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
).db


class DBReadQueryInput(BaseModel):
    query: str = Field(
        description="The SQL query to execute. Only read-only (SELECT) queries are allowed."
    )


class DBReadWriteQueryInput(BaseModel):
    query: str = Field(
        description="The SQL query to execute. Can be a SELECT, INSERT, UPDATE, DELETE, or other statement."
    )


def sql_read_sync(query: str) -> str:
    """Sync SQL read logic (SELECT-only)."""
    try:
        if query.strip().split()[0].lower() in (
            "insert",
            "update",
            "delete",
            "create",
            "drop",
            "alter",
            "truncate",
            "replace",
        ):
            return "Error: Only read-only (SELECT) queries are allowed with this tool."
        result = db.run(query)
        return result
    except Exception as e:
        return f"Error: {str(e)}"


def sql_rw_sync(query: str) -> str:
    """Sync SQL read/write logic."""
    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {str(e)}"


# === SEARCH TOOL ===
serper_search = GoogleSerperAPIWrapper()


class SearchInput(BaseModel):
    query: str = Field(description="The search query string.")


def search_tool_sync(query: str) -> str:
    """Sync logic for searching the web."""
    result = serper_search.run(query)
    return result


# ===== Image Generation Tool =====
image_gen_tool = ImageGenTool(
    api_key=os.environ.get("DALLE_API_KEY"),
    api_base=os.environ.get("DALLE_API_BASE"),
    api_version=os.environ.get("DALLE_API_VERSION"),
    deployment_name=os.environ.get("DALLE_DEPLOYMENT_NAME"),
)


class ImagegenInput(BaseModel):
    prompt: str = Field(description="The prompt used for the image generation.")


def imagegen_tool_sync(prompt: str) -> str:
    try:
        return image_gen_tool.run(prompt)
    except Exception as e:
        return f"Error using ImageGenTool: {str(e)}"















# ===== Wikipedia Tool =====
class WikiInput(BaseModel):
    query: str = Field(description="The search query string for Wikipedia scraping.")


def wiki_tool_sync(query: str) -> str:
    try:
        return wikipedia.summary(query, sentences=1)
    except Exception:
        return "No relevant Wikipedia summary found."


# ===== Random Number Generator Tool =====
class RandomInput(BaseModel):
    random_type: str = Field(
        default="uniform",
        description="Type of randomness: 'uniform', 'gaussian', or 'exponential'. Default is 'uniform'.",
    )
    params: List[float] = Field(
        default_factory=list,
        description=(
            "List of parameters for the distribution:\n"
            "- uniform: [] -> [0,1], [x] -> [x,x+1], [x,y] -> [x,y]\n"
            "- gaussian: [] -> mean=0 std=1, [x] -> mean=x std=1, [x,y] -> mean=x std=y\n"
            "- exponential: [] -> scale=1, [x] -> scale=x"
        ),
    )


def generate_random_sync(random_type: str, params: List[float]) -> float:
    random_type = random_type.lower()

    if random_type == "uniform":
        if len(params) == 0:
            low, high = 0.0, 1.0
        elif len(params) == 1:
            low, high = float(params[0]), float(params[0]) + 1.0
        else:
            low, high = float(params[0]), float(params[1])
        return float(random.uniform(low, high))

    elif random_type == "gaussian":
        if len(params) == 0:
            mean, std = 0.0, 1.0
        elif len(params) == 1:
            mean, std = float(params[0]), 1.0
        else:
            mean, std = float(params[0]), float(params[1])
        return float(random.gauss(mean, std))

    elif random_type == "exponential":
        scale = float(params[0]) if len(params) >= 1 else 1.0
        return float(np.random.exponential(scale))

    else:
        raise ValueError(f"Unknown random_type: {random_type}")


# ===== Hash Generator Tool =====
class HashInput(BaseModel):
    text: str = Field(description="The text to hash.")
    algorithm: str = Field(
        default="sha256",
        description="Hash algorithm: 'sha256', 'md5', or 'sha1'. Default is 'sha256'.",
    )


def hash_string_sync(text: str, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    h.update(text.encode())
    return h.hexdigest()


# ===== URL Screenshot Tool (Playwright) =====
class ScreenshotInput(BaseModel):
    url: Optional[str] = Field(
        default="https://arxiv.org/abs/2505.06934",
        description="The URL of the webpage to capture. Defaults to a sample arXiv PDF.",
    )
    output_file: str = Field(
        default="screenshot.png", description="Output file name for the screenshot."
    )


def screenshot_sync(
    url: str = "https://arxiv.org/abs/2505.06934", output_file: str = "screenshot.png"
) -> str:
    """
    Capture a screenshot of a webpage using Playwright (headless browser).
    Returns the file path to the screenshot.
    """
    with sync_playwright() as p:
        # Launch Firefox for full open-source stack
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")  # wait until network is mostly idle
        page.screenshot(path=output_file, full_page=True)
        browser.close()
    return output_file


# ===== PDF Metadata Tool =====
class PDFMetaInput(BaseModel):
    file_path: str = Field(description="Path to the PDF file.")


def pdf_metadata_sync(file_path: str) -> dict:
    try:
        reader = PdfReader(file_path)
        metadata = reader.metadata or {}
        return {
            "title": metadata.get("/Title", ""),
            "author": metadata.get("/Author", ""),
            "pages": len(reader.pages),
        }
    except Exception as e:
        return {"error": str(e)}


# ===== PDF Summary Tool =====
class PDFSummaryInput(BaseModel):
    file_path: str = Field(description="Path to the PDF file to summarize.")
    max_chars: int = Field(
        default=500, description="Maximum characters to return as a preview snippet."
    )


def pdf_summary_sync(file_path: str, max_chars: int = 500) -> dict:
    try:
        reader = PdfReader(file_path)
        if len(reader.pages) == 0:
            return {"summary": ""}

        # Extract text from first page
        first_page_text = reader.pages[0].extract_text() or ""
        preview = first_page_text.strip()[:max_chars]
        return {
            "summary_preview": preview
            + ("..." if len(first_page_text) > max_chars else "")
        }
    except Exception as e:
        return {"error": str(e)}


# ==== Memory Pydantic Models ===
class RememberInput(BaseModel):
    user_id: str = Field(description="The ID of the user.")
    note: str = Field(description="The note to remember.")


class RecallInput(BaseModel):
    user_id: str = Field(description="The ID of the user.")
    query: Optional[str] = Field(
        default=None,
        description="Optional search query to filter memories. If not provided, returns recent notes.",
    )
