from transformers import pipeline
class ZeroShotClassifier:
    def create_model(self, model_name):
        classifier = pipeline("zero-shot-classification", model=model_name)
        return classifier