from fastapi import FastAPI
from app.routes import router

app = FastAPI()

# Include the router in the app
app.include_router(router)

# Run the app with uvicorn
# welcome route
@app.get("/")
def welcome():
    return {"message": "Welcome to the SVM Classifier API!"}