from transformers import pipeline, BartTokenizer, BartForSequenceClassification
class ZeroShotClassifier:

    def __init__(self, model_name):
        self.model = self.create_model(model_name)
    def create_model(self, model_name):
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForSequenceClassification.from_pretrained(model_name)
        classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
        return classifier