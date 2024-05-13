import configparser

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.SentimentAndIntentionAnalysis import ZeroShotClassifier
from src.data_loader import *

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:8501"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],)
#Get model_name, data_path and labels_path
model_name, data_path, labels_path = get_config()

# Load sentiment labels and intention labels
sentiment_labels, intention_labels = load_labels(labels_path)
# Create Analzer
analyzer = ZeroShotClassifier(model_name=model_name, sentiment_labels=sentiment_labels, intention_labels=intention_labels)

class AnalysisResult(BaseModel):
    sentiment: str
    intention: str

class Text(BaseModel):
    text: str

@app.post("/analyze/")
def analyze_text(data: Text):
    result = analyzer.analyze_text(data.text)
    return AnalysisResult(sentiment=result["sentiment"], intention=result["intention"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api-version:app", host="0.0.0.0", port=8000, reload=True)


