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
        # Sentiment analysis
        sentiment_result = self.model(text, self.sentiment_labels)
        sentiment = sentiment_result["labels"][0]

        # Intention analysis
        intention_result = self.model(text, self.intention_labels)
        intention = intention_result["labels"][0]

        return {"sentiment": sentiment, "intention": intention}