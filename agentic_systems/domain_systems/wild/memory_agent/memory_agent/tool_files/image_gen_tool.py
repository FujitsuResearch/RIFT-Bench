# tools/image_gen_tool.py
import os
import requests
import json
import time
import base64
from typing import Optional, ClassVar
from PIL import Image
from io import BytesIO
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class ImageGenInput(BaseModel):
    """Inputs for Azure OpenAI image generation."""

    prompt: str = Field(..., description="Prompt to generate the image from.")
    size: str = Field(
        default="1024x1024",
        description="Size of the image. Options: 1024x1024, 1792x1024, 1024x1792.",
    )


class ImageGenTool(BaseTool):
    """Tool for generating images using Azure OpenAI (gpt-image-1 or DALL-E 3)."""

    name: ClassVar[str] = "image_generator"
    description: ClassVar[str] = (
        "Generate images using Azure OpenAI image generation. "
        "Provide a detailed prompt describing the image you want to create."
    )
    args_schema: ClassVar[type[BaseModel]] = ImageGenInput

    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None

    def __init__(
        self, api_key=None, api_base=None, api_version=None, deployment_name=None
    ):
        """
        Initialize the image generation tool with Azure OpenAI parameters.

        Args:
            api_key (str, optional): Azure API key. Defaults to DALLE_API_KEY env var.
            api_base (str, optional): Azure endpoint base URL. Defaults to DALLE_API_BASE env var.
            api_version (str, optional): Azure API version. Defaults to DALLE_API_VERSION env var.
            deployment_name (str, optional): Deployment name (e.g. "gpt-image-1"). Defaults to DALLE_DEPLOYMENT_NAME env var.
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("IMAGE_GEN_API_KEY")
        self.api_base = api_base or os.environ.get("IMAGE_GEN_API_BASE")
        self.api_version = api_version or os.environ.get("IMAGE_GEN_API_VERSION")
        self.deployment_name = deployment_name or os.environ.get("IMAGE_GEN_DEPLOYMENT_NAME")

    def _save_image(self, item: dict, save_dir: str) -> Optional[str]:
        """
        Save an image from either a base64 payload (gpt-image-1) or a URL (DALL-E 2/3).

        Returns the saved file path, or None if neither format is present.
        """
        b64_data = item.get("b64_json")
        image_url = item.get("url")

        timestamp = int(time.time())
        image_path = os.path.join(save_dir, f"generated_image_{timestamp}.png")

        if b64_data:
            # gpt-image-1 returns base64-encoded image bytes
            image = Image.open(BytesIO(base64.b64decode(b64_data)))
            image.save(image_path)
            return image_path
        elif image_url:
            # DALL-E 2/3 returns a temporary download URL
            image_response = requests.get(image_url)
            image = Image.open(BytesIO(image_response.content))
            image.save(image_path)
            return image_path

        return None

    def _run(self, prompt: str, size: str = "1024x1024") -> str:
        """
        Generate an image and return the local file path.

        Args:
            prompt (str): Description of the image to generate.
            size (str): Image dimensions, default is 1024x1024.

        Returns:
            str: Path to the saved image, or an error message.
        """
        try:
            save_dir = os.path.join(os.getcwd(), "generated_images")
            os.makedirs(save_dir, exist_ok=True)

            if not all([self.api_key, self.api_base, self.api_version]):
                return (
                    "Error: Azure OpenAI image generation credentials not found. "
                    "Please set DALLE_API_KEY, DALLE_API_BASE, and DALLE_API_VERSION."
                )

            if self.deployment_name:
                endpoint = (
                    f"{self.api_base}/openai/deployments/{self.deployment_name}"
                    f"/images/generations?api-version={self.api_version}"
                )
            else:
                endpoint = (
                    f"{self.api_base}/openai/images/generations"
                    f"?api-version={self.api_version}"
                )

            print(f"Using image generation endpoint: {endpoint}")

            headers = {"Content-Type": "application/json", "api-key": self.api_key}
            data = {"prompt": prompt, "n": 1, "size": size}

            print(f"Sending image generation request with prompt: {prompt[:50]}...")
            response = requests.post(endpoint, headers=headers, json=data)
            print(f"Image generation API response status: {response.status_code}")

            if response.status_code == 200:  # Synchronous response
                response_data = response.json()
                if "data" in response_data and len(response_data["data"]) > 0:
                    image_path = self._save_image(response_data["data"][0], save_dir)
                    if image_path:
                        return image_path

            elif response.status_code == 202:  # Asynchronous operation
                operation_id = response.headers.get("Operation-Location", "").split("/")[-1]
                status_endpoint = (
                    f"{self.api_base}/openai/operations/images/{operation_id}"
                    f"?api-version={self.api_version}"
                )
                print(f"Polling operation at: {status_endpoint}")

                retry_delay = 1
                for _ in range(60):
                    time.sleep(retry_delay)
                    status_response = requests.get(
                        status_endpoint, headers={"api-key": self.api_key}
                    )
                    status_data = status_response.json()

                    if status_data.get("status") == "succeeded":
                        item = status_data.get("result", {}).get("data", [{}])[0]
                        image_path = self._save_image(item, save_dir)
                        if image_path:
                            return image_path

                    if status_data.get("status") == "failed":
                        return f"Image generation failed: {status_data.get('error', {}).get('message')}"

                    retry_delay = min(retry_delay * 1.5, 10)

                return "Image generation timed out"

            else:
                error_message = f"API request failed: {response.status_code}"
                try:
                    error_message += f" - {json.dumps(response.json())}"
                except Exception:
                    error_message += f" - {response.text}"
                print(f"Image generation API error: {error_message}")
                return error_message

        except Exception as e:
            print(f"Exception in image generation: {str(e)}")
            return f"Error generating image: {str(e)}"

    def _arun(self, prompt: str, size: str = "1024x1024") -> str:  # noqa: ARG002
        raise NotImplementedError("Async image generation not implemented")
