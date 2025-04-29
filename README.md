Customer support Chatbot Application
This application is a Flask-based chatbot that processes user queries by interacting with a machine learning model and performing sentiment analysis using the Hugging Face API. It also fetches data from a MongoDB database to provide responses based on common queries.

Features
Sentiment Analysis: Sentiment analysis of user queries using the Hugging Face API (distilbert-base-uncased-finetuned-sst-2-english).

Query Resolution: Responds to user queries by searching a pre-trained decision tree model based on historical data from MongoDB.

MongoDB Integration: Fetches common queries and their resolutions from a MongoDB database.

Model Training: A decision tree model is trained using the TF-IDF vectorized form of user queries to predict appropriate responses.

