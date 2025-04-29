from flask import Flask, render_template, request, jsonify
import pandas as pd
import spacy
import requests

from spacy.cli import download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from pymongo import MongoClient

app = Flask(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading the model...")
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# MongoDB setup
mongodb_uri = "mongodb+srv://vgugan16:gugan2004@cluster0.qyh1fuo.mongodb.net/golang?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongodb_uri)
db = client['golang']
collection = db['customer']

# Fetch data from MongoDB
try:
    cursor = collection.find({}, {"_id": 0, "Query_Text": 1, "Common_Resolution": 1})
    common_queries_list = list(cursor)
    common_queries_df = pd.DataFrame(common_queries_list)
    print("Data loaded from MongoDB successfully.")
except Exception as e:
    print(f"Error loading data from MongoDB: {e}")
    common_queries_df = pd.DataFrame(columns=["Query_Text", "Common_Resolution"])

# Train Decision Tree model
vectorizer = TfidfVectorizer()
model = DecisionTreeClassifier()

try:
    queries = common_queries_df['Query_Text'].astype(str)
    responses = common_queries_df['Common_Resolution'].astype(str)
    X = vectorizer.fit_transform(queries)
    model.fit(X, responses)
    print("Model trained successfully.")
except Exception as e:
    print(f"Model training failed: {e}")
import os
# Hugging Face API details
hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")

# Define the API URL and headers
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
headers = {"Authorization": f"Bearer {hugging_face_token}"}

def query_huggingface(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Sentiment, POS, and NER analysis

import requests
hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")

# Define the API URL and headers
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
headers = {"Authorization": f"Bearer {hugging_face_token}"}
def query_huggingface(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
def analyze_text(user_query):
    try:
        output = query_huggingface({"inputs": user_query})
        print("Raw Hugging Face Output:", output)

        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], list):
            predictions = output[0]
            # Find the prediction with the highest score
            top_prediction = max(predictions, key=lambda x: x["score"])
            result = {
                "sentiment": top_prediction["label"],
                "score": top_prediction["score"]
            }
            print(f"Sentiment Analysis Result: {result}")
            return result
        else:
            result = {
                "sentiment": "ERROR",
                "score": 0.0
            }
            print(f"Sentiment Analysis Result: {result}")
            return result
    except Exception as e:
        print(f"Hugging Face API error: {e}")
        result = {
            "sentiment": "API_ERROR",
            "score": 0.0
        }
        print(f"Sentiment Analysis Result: {result}")
        return result


# Prediction logic
def get_response(user_query):
    user_query_clean = user_query.lower().strip()
    X_input = vectorizer.transform([user_query_clean])
    prediction = model.predict(X_input)[0]

    # Replace Flipkart with DGP
    prediction = prediction.replace('Flipkart', 'DGP').replace('flipkart', 'DGP')

    # Analyze user query (not prediction)
    analysis = analyze_text(user_query)

    return {
        "reply": prediction,
        "analysis": analysis
    }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-response', methods=['POST'])
def chatbot_response():
    user_query = request.form['query']
    result = get_response(user_query)
    
    # Include the sentiment in the response
    result['analysis']['sentiment'] = result['analysis'].get("sentiment", "ERROR")
    result['analysis']['score'] = result['analysis'].get("score", 0.0)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
