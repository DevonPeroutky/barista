from io import BytesIO

from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.inference.constants import AUGMENTED_MODEL_PATH
from src.inference.service import InferenceService
from src.inference.utils import get_model_name_from_path

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://localhost:5173"
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):

    # Read the file content
    file_content = await file.read()

    # Convert the file content to a PIL image
    pil_image = Image.open(BytesIO(file_content))

    # Print out the image size
    print(pil_image.size)

    response = InferenceService(model_path=AUGMENTED_MODEL_PATH).generate_caption(pil_image)

    return {"roast_response": response}
