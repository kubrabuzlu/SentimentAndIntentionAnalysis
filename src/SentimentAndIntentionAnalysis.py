from transformers import pipeline
class ZeroShotClassifier:

    def __init__(self, model_name):
        self.model = self.create_model(model_name)
    def create_model(self, model_name):
        classifier = pipeline("zero-shot-classification", model=model_name)
        return classifier