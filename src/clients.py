import time
from io import BytesIO

import requests
from abc import ABC, abstractmethod

import openai
import os
import anthropic
import PIL

from PIL import Image

from openai import OpenAI

from src.utils import download_and_encode_image, pil_to_base64, resize_with_aspect_ratio

# Read in OPENAI_API_KEY from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")


class VisionAssistant(ABC):
    client = None

    def __init__(self):
        pass

    @abstractmethod
    def describe_person_in_image(self, image_url, retries=3, delay=60):
        pass

    @staticmethod
    def download_image(image_url) -> str:
        image_base64 = download_and_encode_image(image_url)
        return image_base64

    @staticmethod
    def download_image_as_pil(image_url: str) -> Image:
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return image
        else:
            print(f"Error: Failed to download image from {image_url}")
            return None


class OpenAIVisionAssistant(VisionAssistant):

    def __init__(self, api_key=OPENAI_API_KEY):
        super().__init__()
        self.client = OpenAI(api_key=api_key)

    def describe_person_in_image(self, image_url, model="gpt-4-vision-preview", retries=3, delay=90):
        print("Using OpenAI Vision API to describe person in image from URL: ", image_url)
        image = self.download_image_as_pil(image_url)
        if image is None:
            raise ValueError("Failed to download image from URL", image_url)

        # resize image to have width of 1024 and height of auto
        image = resize_with_aspect_ratio(image, new_width=1024)

        # convert image to base64 url
        base64_url = pil_to_base64(image=image)

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a highly capable vision model. Describe person in image with as much detail as possible. Include age, race, hair color, weight, clothing, hair style, accessories, facial features, accessories, and any unusual features. Be very detailed about facial structure and facial features. No intro or extra fluff."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe this person in the image, be detailed"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_url}",
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                )
                if response.choices[0] == "I'm sorry, but I can't provide assistance with that request.":
                    raise ValueError("I'm sorry, but I can't provide assistance with that request.", image_url)
                return response.choices[0].message.content if response and response.choices else None
            except openai.RateLimitError as e:
                if attempt < retries - 1:
                    print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print("Rate limit exceeded. Maximum retries reached.")
                    raise e
            except openai.BadRequestError as e:
                print(f"URL: {image_url}")
                print(f"Error: {e}")
                raise e


class ClaudeVisionAssistant(VisionAssistant):

    def __init__(self, api_key=CLAUDE_API_KEY, model: str = "claude-3-haiku-20240307"):
        super().__init__()
        self.client = anthropic.Anthropic(
            api_key=api_key,
        )
        self.model = model

    def describe_person_in_image(self, image_url, retries=3, delay=60):
        print("Using Claude Vision API to describe person in image from URL: ", image_url)

        # Get base url of image_url
        image_media_url = image_url.split("?")[0]

        # derive media_type from image_url
        if image_media_url.endswith(".png"):
            image2_media_type = "image/png"
            format = "PNG"
        elif image_media_url.endswith(".jpeg") or image_media_url.endswith(".jpg"):
            image2_media_type = "image/jpeg"
            format = "JPEG"
        elif image_media_url.endswith(".webp"):
            image2_media_type = "image/webp"
            format = "WEBP"
        else:
            raise ValueError("Unsupported image format", image_url)

        print("Image is of type: ", image2_media_type)

        image = self.download_image_as_pil(image_url)
        if image is None:
            raise ValueError("Failed to download image from URL", image_url)

        # resize image to have width of 1024 and height of auto
        image = resize_with_aspect_ratio(image, new_width=1024)

        # convert image to base64 url
        base64_url = pil_to_base64(image=image, format=format)

        for attempt in range(retries):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=0,
                    system="You are a highly capable vision model. Describe the person in the image with as much detail as possible, including age, race, hair color, weight, clothing, hair style, accessories, facial features, accessories, and any unusual features. You are very detailed about facial structure and facial features. No introduction or extra fluff",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": image2_media_type,
                                        "data": base64_url
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": "This person wants to be described in detail for an experiment. Describe this person and focus on their hair, face, and clothing"
                                }
                            ]
                        }
                    ]
                )
                return message.content[0].text
            except anthropic.RateLimitError as e:
                if attempt < retries - 1:
                    print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print("Rate limit exceeded. Maximum retries reached.")
                    raise e

        return None