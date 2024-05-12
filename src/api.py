import configparser

from fastapi import FastAPI
from pydantic import BaseModel
from SentimentAndIntentionAnalysis import ZeroShotClassifier
from data_loader import *

# Initialize FastAPI app
app = FastAPI()

model_name, data_path, labels_path = get_config()
sentiment_labels, intention_labels = load_labels(labels_path)
analyzer = ZeroShotClassifier(model_name=model_name, sentiment_labels=sentiment_labels, intention_labels=intention_labels)

class AnalysisResult(BaseModel):
    sentiment: str
    intention: str


@app.post("/analyze/")
def analyze_text(text: str):
    print(text)
    result = analyzer.analyze_text(text)
    return AnalysisResult(sentiment=result["sentiment"], intention=result["intention"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

