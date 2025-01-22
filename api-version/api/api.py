import configparser

from fastapi import FastAPI
from pydantic import BaseModel
from SentimentAndIntentionAnalysis import ZeroShotClassifier

# Initialize FastAPI app
app = FastAPI()

# Create Analyzer
analyzer = ZeroShotClassifier(model_name='facebook/bart-large-mnli')

class AnalysisResult(BaseModel):
    sentiment: str
    intention: str

class Text(BaseModel):
    text: str

@app.post("/analyze/")
def analyze_text(data: Text):
    result = analyzer.analyze_text(data.text)
    return AnalysisResult(sentiment=result["sentiment"], intention=result["intention"])
