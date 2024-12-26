from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
MODEL_PATH = "model/svm_text_model.pkl"
DATA_PATH = "data/dataset.csv"

def train_model(data_path: str, model_path: str):
    
    #Data
    data = pd.read_csv(data_path)
    x,y = data['text'], data['label']
    
    #Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    #Create pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', SVC(kernel='linear', probability=True))
    ])
    
    
    # Train 
    model.fit(x_train, y_train)
    
    #Test
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(y_pred,x_test)
    #save model
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")
    
if __name__ == "__main__":
    train_model(DATA_PATH, MODEL_PATH)