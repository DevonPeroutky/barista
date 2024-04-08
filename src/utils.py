import hashlib
import base64
from io import BytesIO

import PIL
import requests

from itertools import chain
from typing import Iterable, Callable, TypeVar, List, Optional
from PIL import Image


A = TypeVar('A')
T = TypeVar('T')


def flat_map(func: Callable[[A], Iterable[T]], iterable: Iterable[A]) -> List[T]:
    return list(chain.from_iterable(map(func, iterable)))


def download_and_encode_image(image_url, format="JPEG") -> str:
    # Download the image data
    response = requests.get(image_url)

    # Check if the request was successful
    if response.status_code == 200:

        # Download the image into a PIL and then resize it such that no dimension is greater than 8000 pixels
        image = Image.open(BytesIO(response.content))

        if image.mode != "RGB":
            image = image.convert("RGB")

        max_dimension = 4096
        if image.width > max_dimension or image.height > max_dimension:
            if image.width > image.height:
                new_width = max_dimension
                new_height = int(image.height * (max_dimension / image.width))
            else:
                new_height = max_dimension
                new_width = int(image.width * (max_dimension / image.height))
            image = image.resize((new_width, new_height))

        # Encode the PIL image data as a base64 string
        buffered = BytesIO()

        image.save(buffered, format=format)
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return base64_image
    else:
        print(f"Error: Failed to download image from {image_url}")
        return None


def pil_to_base64(image: Image, format="JPEG") -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")

    buffered = BytesIO()
    image.save(buffered, format=format)
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return base64_image


def resize_with_aspect_ratio(image: Image, new_width: Optional[int] = None, new_height: Optional[int] = None) -> Image:
    assert new_width or new_height, "Either new_width or new_height must be specified"

    if new_width and new_height:
        image = image.resize((new_width, new_height))
    elif new_width and not new_height:
        new_height = int((new_width / image.width) * image.height)
        image = image.resize((new_width, new_height))
    elif not new_width and new_height:
        new_width = int((new_height / image.height) * image.width)
        image = image.resize((new_width, new_height))
    else:
        raise ValueError("Either new_width or new_height must be specified")

    return image


def get_image_hash(image):
    # Convert image to bytes
    byte_arr = BytesIO()
    image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()

    # Compute hash
    hasher = hashlib.sha256()
    hasher.update(byte_arr)
    return hasher.hexdigest()