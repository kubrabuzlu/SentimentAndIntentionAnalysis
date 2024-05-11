import configparser
import json
from SentimentAndIntentionAnalysis import ZeroShotClassifier
from transformers import pipeline
print("Sentiment and Intention Analysis Project")

# Sentiment Labels
sentiment_labels = ["positive", "negative", "neutral"]

# Intention Labels
intention_labels = ["Inquire","Information Gathering", "Payment", "Trade", "Customer Dissatisfaction",
                        "Customer Satisfaction", "Closing Remarks", "Change Package", "Upgrade", "Purchase"]

def get_config():

    # Read config
    config = configparser.ConfigParser()
    config.read(r"C:\Users\kbuzlu\PycharmProjects\SentimentIntentionAnalysis\config\config.ini")

    # Get model_name
    model_name = config["model"]["name"]

    # Get data path
    data_path = config["data"]["data_dir"]
    return model_name, data_path

def load_data(data_path):
    with open(data_path, "r") as file:
        data = json.load(file)
    texts = [entry["text"] for entry in data]
    return texts

if __name__ == "__main__":
    model_name, data_path = get_config()
    data = load_data(data_path)
    create_classifier = ZeroShotClassifier(model_name)
    classifier = create_classifier.model
    for sentence in data:
        sentiment_result = classifier(sentence, sentiment_labels)
        print(f"\nSentence: {sentence}")
        print("Sentiment:", sentiment_result["labels"][0])

        # Intention analysis
        intention_result = classifier(sentence, intention_labels)
        print("Intention:", intention_result["labels"][0])