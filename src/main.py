from SentimentAndIntentionAnalysis import ZeroShotClassifier
from data_loader import get_config, load_labels, load_data
import json

def main():
    # Get path and potential labels path config.ini
    model_name, data_path, labels_path = get_config()
    sentiment_labels, intention_labels = load_labels(labels_path)
    # Load data
    print("Data is loading...")
    data = load_data(data_path)
    # Create model
    print("Model is creating...")
    classifier = ZeroShotClassifier(model_name, sentiment_labels, intention_labels)

    # Analyze data, print results and save
    results = []
    for sentence in data:
        result = classifier.analyze_text(sentence)
        result_dict = {
            "sentence": sentence,
            "sentiment": result["sentiment"],
            "intention": result["intention"]
        }
        results.append(result_dict)
        print(f'Target sentences:{sentence}')
        print(f'Sentiment:{result["sentiment"]}, Intention:{result["intention"]}\n')

    with open('../data/results.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print("Results saved to 'results.json' file.")

if __name__ == "__main__":
    main()