import joblib
from model import train_spam_classification
import os

class TextModelPredictor:
    def __init__(self, model_path: str):
        if model_path is None:
            raise ValueError("Model path is required")
        elif os.path.exists(model_path) is False:
            train_spam_classification.train_model("data/dataset.csv", model_path)
        self.model = joblib.load(model_path)
        
        
    def predict(self, text: str):
        prediction = self.model.predict([text])[0]
        probability = self.model.predict_proba([text])[0]
        return {"prediction": prediction, "probability": probability.tolist()}

class ImageModelPredictor:
    def __init__(self, model_path: str):
        if model_path is None:
            raise ValueError("Model path is required")
        elif os.path.exists(model_path) is False:
            train_spam_classification.train_model("data/images", model_path)
        self.model = joblib.load(model_path)
        
        
    def predict(self, image: str):
        prediction = self.model.predict([image])[0]
        probability = self.model.predict_proba([image])[0]
        return {"prediction": prediction, "probability": probability.tolist()}