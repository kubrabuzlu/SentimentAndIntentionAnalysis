import configparser
import json
import os


def get_config():
    # Read config
    config = configparser.ConfigParser()
    config.read(os.path.join("..", "config", "config.ini"))

    # Get model_name
    model_name = config["model"]["name"]

    # Get data path
    data_path = config["data"]["data_dir"]

    # Get possible labels
    labels_path = config["data"]["possible_labels_dir"]
    return model_name, data_path, labels_path


def load_data(data_path):
    # Normalize the path for cross-platform compatibility
    normalized_data_path = os.path.normpath(data_path)

    # Load data
    with open(data_path, "r") as file:
        data = json.load(file)
    texts = [entry["text"] for entry in data]
    return texts


def load_labels(labels_path):
    # Load labels
    with open(labels_path, "r") as file:
        data = json.load(file)
    sentiment_labels = data["sentiment_labels"]
    intention_labels = data["intention_labels"]
    return sentiment_labels, intention_labels
