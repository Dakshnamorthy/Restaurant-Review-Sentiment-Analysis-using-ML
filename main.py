import os
import re
import json
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

# Initialize NLTK
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)

app = FastAPI(
    title="Restaurant Review Sentiment Analysis API",
    description="API to predict sentiment of restaurant reviews and explore model performance.",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

# Global model state
MODEL = None
VECTORIZER = None
METRICS = None
VOCAB_COEFS = {}
REVIEWS_DF = None

# Model Loading
@app.on_event("startup")
def load_assets():
    global MODEL, VECTORIZER, METRICS, VOCAB_COEFS, REVIEWS_DF
    
    # Load model and vectorizer
    if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
        MODEL = joblib.load("model.pkl")
        VECTORIZER = joblib.load("vectorizer.pkl")
        
        # Build dictionary of vocabulary coefficients for term highlighting
        feature_names = VECTORIZER.get_feature_names_out()
        coefs = MODEL.coef_[0]
        VOCAB_COEFS = dict(zip(feature_names, coefs))
        print("Model and vectorizer loaded successfully.")
    else:
        print("Warning: model.pkl or vectorizer.pkl not found! Please run train_model.py first.")
        
    # Load model metrics
    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json", "r") as f:
            METRICS = json.load(f)
        print("Metrics loaded successfully.")
    else:
        print("Warning: model_metrics.json not found! Please run train_model.py first.")
        
    # Load reviews for explorer
    if os.path.exists("Reviews.tsv"):
        REVIEWS_DF = pd.read_csv("Reviews.tsv", delimiter="\t")
        print(f"Reviews dataset loaded. Total reviews: {len(REVIEWS_DF)}")
    else:
        print("Warning: Reviews.tsv not found.")

# Text Preprocessing helper
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_and_lemmatize(text: str) -> str:
    # Remove special characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    words = text.split()
    # Lemmatize & stop words
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# Models
class ReviewInput(BaseModel):
    text: str

class BatchReviewsInput(BaseModel):
    reviews: List[str]

# API Endpoints

@app.post("/api/predict")
def predict_sentiment(input_data: ReviewInput):
    if MODEL is None or VECTORIZER is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please train the model.")
        
    raw_text = input_data.text
    if not raw_text.strip():
        return {
            "sentiment": "Neutral",
            "confidence": 1.0,
            "cleaned_text": "",
            "probability_positive": 0.5,
            "probability_negative": 0.5,
            "word_contributions": []
        }
        
    # Preprocess text
    cleaned_text = clean_and_lemmatize(raw_text)
    
    # Vectorize
    vector = VECTORIZER.transform([cleaned_text]).toarray()
    
    # Predict
    prediction = int(MODEL.predict(vector)[0])
    probabilities = MODEL.predict_proba(vector)[0]  # [prob_neg, prob_pos]
    
    prob_neg = float(probabilities[0])
    prob_pos = float(probabilities[1])
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = prob_pos if prediction == 1 else prob_neg
    
    # Find word contributions (which words in the review contributed and by how much)
    # Splitting original text into words for mapping
    # We want to map original lowercase words to coefficients if they are in the vocabulary
    words_original = re.sub('[^a-zA-Z]', ' ', raw_text).lower().split()
    word_contributions = []
    
    for word in words_original:
        lemmatized_word = lemmatizer.lemmatize(word)
        # Check if the word or its lemmatized form is in the coefficients dictionary
        weight = 0.0
        found = False
        
        if word in VOCAB_COEFS:
            weight = float(VOCAB_COEFS[word])
            found = True
        elif lemmatized_word in VOCAB_COEFS:
            weight = float(VOCAB_COEFS[lemmatized_word])
            found = True
            
        if found:
            word_contributions.append({
                "word": word,
                "weight": weight
            })
            
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "probability_positive": prob_pos,
        "probability_negative": prob_neg,
        "cleaned_text": cleaned_text,
        "word_contributions": word_contributions
    }

@app.post("/api/predict-batch")
def predict_sentiment_batch(input_data: BatchReviewsInput):
    if MODEL is None or VECTORIZER is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please train the model.")
        
    reviews = input_data.reviews
    results = []
    pos_count = 0
    neg_count = 0
    
    for review in reviews:
        if not review.strip():
            continue
            
        cleaned = clean_and_lemmatize(review)
        vector = VECTORIZER.transform([cleaned]).toarray()
        prediction = int(MODEL.predict(vector)[0])
        probabilities = MODEL.predict_proba(vector)[0]
        
        prob_neg = float(probabilities[0])
        prob_pos = float(probabilities[1])
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = prob_pos if prediction == 1 else prob_neg
        
        if prediction == 1:
            pos_count += 1
        else:
            neg_count += 1
            
        results.append({
            "review": review,
            "sentiment": sentiment,
            "confidence": confidence
        })
        
    total = len(results)
    
    return {
        "results": results,
        "summary": {
            "total": total,
            "positive": pos_count,
            "negative": neg_count,
            "pos_percentage": float((pos_count / total) * 100) if total > 0 else 0,
            "neg_percentage": float((neg_count / total) * 100) if total > 0 else 0
        }
    }

@app.get("/api/metrics")
def get_metrics():
    if METRICS is None:
        raise HTTPException(status_code=503, detail="Metrics are not loaded.")
    return METRICS

@app.get("/api/reviews")
def get_reviews(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: str = Query(""),
    sentiment: str = Query("all") # "all", "positive", "negative"
):
    if REVIEWS_DF is None:
        raise HTTPException(status_code=503, detail="Reviews dataset not loaded.")
        
    filtered_df = REVIEWS_DF.copy()
    
    # Filter by sentiment
    if sentiment == "positive":
        filtered_df = filtered_df[filtered_df['Liked'] == 1]
    elif sentiment == "negative":
        filtered_df = filtered_df[filtered_df['Liked'] == 0]
        
    # Filter by search
    if search:
        filtered_df = filtered_df[filtered_df['Review'].str.contains(search, case=False, na=False)]
        
    total_items = len(filtered_df)
    total_pages = (total_items + limit - 1) // limit
    
    # Paginate
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_df = filtered_df.iloc[start_idx:end_idx]
    
    reviews_list = paginated_df.to_dict(orient="records")
    
    return {
        "reviews": reviews_list,
        "total": total_items,
        "page": page,
        "limit": limit,
        "pages": max(1, total_pages)
    }

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")
