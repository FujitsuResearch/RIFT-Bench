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
from tool_files.custom_tools import *
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

llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        model=os.getenv("AZURE_MODEL_NAME"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        temperature=0.0,
    )



# === SQL TOOLS ===
async def sql_read_tool_async(query: str) -> str:
    return await asyncio.to_thread(sql_read_sync, query)


sql_read_tool = Tool(
    name="sql_read_tool",
    description="Useful for reading data from a SQL database.",
    args_schema=DBReadQueryInput,
    func=sql_read_sync,
    coroutine=sql_read_tool_async,
)


async def sql_read_write_tool_async(query: str) -> str:
    return await asyncio.to_thread(sql_rw_sync, query)


sql_read_write_tool = Tool(
    name="sql_read_write_tool",
    description="Useful for reading and writing data to a SQL database.",
    args_schema=DBReadWriteQueryInput,
    func=sql_rw_sync,
    coroutine=sql_read_write_tool_async,
)


# === SEARCH TOOL ===
async def search_tool_async(query: str) -> str:
    return await asyncio.to_thread(search_tool_sync, query)


search_tool = Tool(
    name="search_tool",
    description="Useful for searching information on the web.",
    args_schema=SearchInput,
    func=search_tool_sync,
    coroutine=search_tool_async,
)


# ===== Image Generation Tool =====
async def imagegen_tool_async(prompt: str) -> str:
    return await asyncio.to_thread(imagegen_tool_sync, prompt)


imagegen_tool = Tool(
    name="imagegen_tool",
    description="Useful for generating images from text.",
    args_schema=ImagegenInput,
    func=imagegen_tool_sync,
    coroutine=imagegen_tool_async,
)
















# ===== Wikipedia Tool =====
async def wiki_tool_async(query: str) -> str:
    return await asyncio.to_thread(wiki_tool_sync, query)


wiki_tool = Tool(
    name="wiki_tool",
    description="Useful for scraping content from Wikipedia.",
    args_schema=WikiInput,
    func=wiki_tool_sync,
    coroutine=wiki_tool_async,
)


# ===== Random Number Generator Tool =====
async def generate_random_async(random_type: str, params: List[float]) -> float:
    return await asyncio.to_thread(generate_random_sync, random_type, params)


random_tool = StructuredTool(
    name="random_tool",
    description="Generates a random number using uniform, gaussian, or exponential distribution.",
    args_schema=RandomInput,
    func=generate_random_sync,
    coroutine=generate_random_async,
)


# ===== Hash Generator Tool =====
async def hash_string_async(text: str, algorithm: str = "sha256") -> str:
    return await asyncio.to_thread(hash_string_sync, text, algorithm)


hash_tool = StructuredTool(
    name="hash_tool",
    description="Computes a cryptographic hash (SHA256, MD5, SHA1) of a string.",
    args_schema=HashInput,
    func=hash_string_sync,
    coroutine=hash_string_async,
)


# ===== URL Screenshot Tool (Playwright) =====
async def screenshot_async(
    url: str = "https://arxiv.org/abs/2505.06934", output_file: str = "screenshot.png"
) -> str:
    """
    Async wrapper for Playwright screenshot function.
    """
    return await asyncio.to_thread(screenshot_sync, url, output_file)


screenshot_tool = StructuredTool(
    name="screenshot_tool",
    description="Takes a full-page screenshot of a webpage using a headless browser. Can be called with no url address (has default option)",
    args_schema=ScreenshotInput,
    func=screenshot_sync,
    coroutine=screenshot_async,
)


# ===== PDF Metadata Tool =====
async def pdf_metadata_async(file_path: str) -> dict:
    return await asyncio.to_thread(pdf_metadata_sync, file_path)


pdf_metadata_tool = StructuredTool(
    name="pdf_metadata_tool",
    description="Extracts title, author, and page count from a PDF file without reading the full text.",
    args_schema=PDFMetaInput,
    func=pdf_metadata_sync,
    coroutine=pdf_metadata_async,
)


# ===== PDF Summary Tool =====
async def pdf_summary_async(file_path: str, max_chars: int = 500) -> dict:
    return await asyncio.to_thread(pdf_summary_sync, file_path, max_chars)


pdf_summary_tool = StructuredTool(
    name="pdf_summary_tool",
    description="Extracts a short preview text from the first page of a PDF, useful for LLM summarization.",
    args_schema=PDFSummaryInput,
    func=pdf_summary_sync,
    coroutine=pdf_summary_async,
)
