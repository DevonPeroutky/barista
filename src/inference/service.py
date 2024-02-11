import time
from typing import Optional

from PIL.Image import Image


class InferenceService:
    model = None

    def generate_caption(self, pil_image: Image, prompt: Optional[str] = None) -> str:
        self.model = None

        # Sleep for a few seconds
        time.sleep(3)

        # Print out the image size
        print(pil_image.size)

        return "TBD"
