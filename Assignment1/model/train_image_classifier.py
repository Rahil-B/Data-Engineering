import os
import cv2
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Constants
IMAGE_SIZE = (64, 64)  # Resize images to 64x64
DATA_PATH = "data/images"
MODEL_PATH = "model/svm_image_model.pkl"

def load_images(data_path):
    images = [] #['cat1.jpeg','cat2.jpeg','cat3.jpeg','dog1.jpeg','dog2.jpeg','dog3.jpeg']
    labels = []#[0,0,0,1,1,1]
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, IMAGE_SIZE)
                images.append(image.flatten())
                labels.append(label)
    return np.array(images), np.array(labels)

def train_model(data_path, model_path):
    # Load and preprocess data
    x, y = load_images(data_path)
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    # Create pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='linear', probability=True))
    ])
    
    # Train
    model.fit(x_train, y_train)
    
    # Test
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    train_model(DATA_PATH, MODEL_PATH)