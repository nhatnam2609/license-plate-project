from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
from PIL import Image
import io

# Initialize the PaddleOCR model
ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, rec_model_dir="output/rec_ppocr_v4/best_accuracy",rec_char_dict_path="paddleocr_training/dict.txt")

# Initialize FastAPI app
app = FastAPI()

@app.post("/ocr")
async def recognize_license_plate(image: UploadFile = File(...)):
    """
    Endpoint to recognize characters from a license plate image.
    Input: Image (uploaded file)
    Output: Recognized characters and confidence scores
    """
    try:
        # Read the uploaded image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Convert PIL image to format usable by PaddleOCR
        img_path = "temp_image.jpg"
        img.save(img_path)

        # Run OCR
        result = ocr.ocr(img_path, cls=False, det=True, rec=True)[0]

        # Parse results
        recognized_texts = [line[0] for line in result]
        confidence_scores = [line[1] for line in result]

        # Prepare the response
        response = {
            "recognized_texts": recognized_texts,
            "confidence_scores": confidence_scores
        }

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)