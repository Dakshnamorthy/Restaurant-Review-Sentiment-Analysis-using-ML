import re
import json
import numpy as np
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_and_export():
    print("Step 1: Downloading NLTK resources...")
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    print("Step 2: Loading dataset Reviews.tsv...")
    data = pd.read_csv("Reviews.tsv", delimiter="\t")
    print(f"Dataset loaded. Total rows: {len(data)}")
    
    print("Step 3: Preprocessing reviews...")
    corpus = []
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Custom stop words modification: we want to keep negatives like 'not', 'no', 'never' 
    # since they affect sentiment heavily, but the standard NLTK stopwords include them.
    # Let's see if we should refine the stop words list. The original notebook just used the standard stop_words.
    # To keep consistency with the notebook, we will use the notebook's exact stop words logic,
    # but we can verify it. Let's stick to the exact logic for now to maintain the model's structure.
    
    for review in data['Review']:
        # Remove special characters
        review = re.sub('[^a-zA-Z]', ' ', str(review))
        # Convert to lowercase
        review = review.lower()
        # Tokenize
        review = review.split()
        # Lemmatization & Stop words removal
        review = [lemmatizer.lemmatize(word) for word in review if word not in stop_words]
        # Re-join
        review = ' '.join(review)
        corpus.append(review)
        
    print("Step 4: Vectorizing using TF-IDF...")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
    X = tfidf.fit_transform(corpus).toarray()
    y = data['Liked']
    
    print("Step 5: Splitting dataset (70% train, 30% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print("Step 6: Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    print("Step 7: Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    print("Step 8: Extracting feature importances (coefficients)...")
    feature_names = tfidf.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Pair feature names with coefficients
    feature_coefs = list(zip(feature_names, coefficients))
    
    # Sort by weight
    feature_coefs_sorted = sorted(feature_coefs, key=lambda x: x[1])
    
    # Top negative words (most negative coefficients)
    top_negative = [{"word": word, "weight": float(coef)} for word, coef in feature_coefs_sorted[:30]]
    
    # Top positive words (most positive coefficients, sorted descending)
    top_positive = [{"word": word, "weight": float(coef)} for word, coef in feature_coefs_sorted[-30:]]
    top_positive.reverse()
    
    print("Step 9: Calculating dataset statistics...")
    total_reviews = len(data)
    positive_reviews = int((data['Liked'] == 1).sum())
    negative_reviews = int((data['Liked'] == 0).sum())
    
    # Calculate average word lengths
    data['word_count'] = data['Review'].apply(lambda x: len(str(x).split()))
    avg_words_pos = float(data[data['Liked'] == 1]['word_count'].mean())
    avg_words_neg = float(data[data['Liked'] == 0]['word_count'].mean())
    
    metrics = {
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
        "classification_report": report,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "dataset_stats": {
            "total": total_reviews,
            "positive": positive_reviews,
            "negative": negative_reviews,
            "pos_percentage": float((positive_reviews / total_reviews) * 100),
            "neg_percentage": float((negative_reviews / total_reviews) * 100),
            "avg_words_pos": avg_words_pos,
            "avg_words_neg": avg_words_neg
        }
    }
    
    print("Step 10: Saving models and metrics...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(tfidf, 'vectorizer.pkl')
    
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Success! Model ('model.pkl'), Vectorizer ('vectorizer.pkl'), and Metrics ('model_metrics.json') have been successfully exported.")

if __name__ == "__main__":
    train_and_export()
