from fastapi import APIRouter, HTTPException, File, UploadFile
from model.predict import TextModelPredictor, ImageModelPredictor, NextWordPredictor
from model.monitor import monitor_prediction_time
import cv2
import numpy as np

router = APIRouter()
text_predictor = TextModelPredictor("model/svm_text_model.pkl")
image_classifier = ImageModelPredictor("model/svm_image_model.pkl")
next_word_predictor = NextWordPredictor("model/svm_text_model.pkl")

@router.get("/text/predict")
@monitor_prediction_time
def predict(text: str):
    try:
        result = text_predictor.predict(text)
        print(result)
        return {"status": "success", "result": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/image/classify")
@monitor_prediction_time
async def img_predict(img: UploadFile = File(...)):
    try:
        print("Reading image file")
        # Read the image file
        contents = await img.read()
        # Convert the image to a numpy array
        nparr = np.fromstring(contents, np.uint8)
        # Decode the image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Preprocess the image if necessary
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64)).flatten()
        # Predict the class
        print("Predicting image class")
        result = image_classifier.predict(contents)
        print(result)
        return {"status": "success", "result": str(result[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/text/predict_next_word")
@monitor_prediction_time
async def text_predict(text: str):
    try:
        print("Predicting next word")
        result = next_word_predictor.predict(text)
        print(result)
        return {"status": "success", "result": result["prediction"], "probability": result["probability"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))