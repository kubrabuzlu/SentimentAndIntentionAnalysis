from transformers import pipeline, BartTokenizer, BartForSequenceClassification

class ZeroShotClassifier:

    def __init__(self, model_name):
        self.model = self.create_model(model_name)
        self.model_name = model_name
        self.sentiment_labels = ["Positive", "Negative", "Neutral"]
        self.intention_labels = ["Inquire", "Inform", "Payment", "Price", "Trade In", "Discount", "Complaint",
                                 "Approve", "Selling", "Confusion", "Change Package", "Upgrade", "Purchase", "Help"]

    def create_model(self, model_name):
        # Create Model
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForSequenceClassification.from_pretrained(model_name)
        classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
        return classifier

    def analyze_text(self, text):
        results = list(self.model(text, self.labels)['labels'])
        i = 0
        sentiment = None
        intention = None
        while (sentiment is None) or (intention is None):
            if results[i] in self.sentiment_labels:
                # Sentiment analyze result
                sentiment = results[i]
            if results[i] in self.intention_labels:
                # Intention analyze result
                intention = results[i]
            i += 1
        return {"sentiment": sentiment, "intention": intention}