import configparser

print("Sentiment and Intention Analysis Project")

# Config dosyasını okuma
config = configparser.ConfigParser()
config.read("config.ini")

# Modelin adını al
model_name = config["model"]["name"]

# Sentiment etiketlerini al
sentiment_labels = ["positive", "negative", "neutral"]

# Intention etiketlerini al
intention_labels = ["Inquire","Information Gathering", "Payment", "Trade", "Customer Dissatisfaction",
                    "Customer Satisfaction", "Closing Remarks", "Change Package", "Upgrade", "Purchase"]

data_path = config["data"]["data_dir"]

