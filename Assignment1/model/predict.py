import joblib
from model import train_spam_classification,train_next_word_model,train_image_classifier
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
            train_image_classifier.train_model("data/images", model_path)
        self.model = joblib.load(model_path)
        
        
    def predict(self, image: str):
        prediction = self.model.predict([image])[0]
        probability = self.model.predict_proba([image])[0]
        return {"prediction": prediction, "probability": probability.tolist()}
    
class NextWordPredictor:
    def __init__(self, model_path: str):
        print("Initiatin Next Word")
        if model_path is None:
            raise ValueError("Model path is required")
        elif not os.path.exists(model_path):
            train_next_word_model.train_model("data/next_word_dataset.csv", model_path)
        self.model = joblib.load(model_path)
        
    def predict(self, text: str):
        # Generate n-grams for the input text
        ngrams = train_next_word_model.generate_ngrams(text)
        ngrams_joined = " ".join(ngrams)
        
        # Predict the next word
        prediction = self.model.predict([ngrams_joined])[0]
        probability = self.model.predict_proba([ngrams_joined])[0]
        
        return {"prediction": prediction, "probability": probability.tolist()}