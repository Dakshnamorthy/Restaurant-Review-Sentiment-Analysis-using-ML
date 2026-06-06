# SentiChef - Restaurant Review Sentiment Analyzer

A machine learning-powered web application that analyzes restaurant reviews and classifies them as positive or negative using a trained Logistic Regression model.

## Features

- **🎯 Sentiment Prediction**: Analyze individual restaurant reviews and get sentiment predictions (Positive/Negative)
- **📊 Batch Processing**: Analyze multiple reviews at once
- **📈 Model Metrics**: View model performance metrics, accuracy, confusion matrix, and classification reports
- **🔍 Review Explorer**: Browse and search through the entire dataset with sentiment filtering
- **💡 Word Contributions**: See which words in a review contribute to the sentiment prediction
- **🎨 Modern UI**: Beautiful, responsive interface (SentiChef) built with vanilla JavaScript and CSS
- **📱 Mobile Friendly**: Access the app on mobile devices via local network
- **⚡ Fast API**: Built with FastAPI for high-performance predictions

## Tech Stack

- **Backend**: FastAPI, Python
- **Machine Learning**: scikit-learn (LogisticRegression, TfidfVectorizer)
- **NLP**: NLTK (Lemmatization, Stop words removal)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Data Processing**: pandas, numpy
- **Model Serialization**: joblib

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "restaurant sentiment analysis"
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   **On Windows (PowerShell):**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   
   **On Windows (Command Prompt):**
   ```cmd
   venv\Scripts\activate.bat
   ```
   
   **On macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install fastapi uvicorn pandas scikit-learn nltk joblib
   ```

## Usage

### Train the Model (Optional)

If you want to retrain the model from scratch:

```bash
python train_model.py
```

This will:
- Load the Reviews.tsv dataset
- Preprocess the reviews (lowercasing, lemmatization, stop word removal)
- Vectorize using TF-IDF with max 3000 features
- Train a Logistic Regression classifier
- Export: `model.pkl`, `vectorizer.pkl`, and `model_metrics.json`

### Run the Application

Start the FastAPI server:

```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Then open your browser and navigate to:
```
http://127.0.0.1:8000/
```

### Access on Mobile (Same Network)

1. Find your PC's local IP address:
   ```powershell
   ipconfig
   ```
   Look for "IPv4 Address" (typically `192.168.x.x`)

2. Start the server on all interfaces:
   ```bash
   python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. On your mobile browser, open:
   ```
   http://<YOUR_PC_IP>:8000/
   ```

## API Endpoints

### 1. Predict Single Review
```http
POST /api/predict
Content-Type: application/json

{
  "text": "This restaurant had amazing food and great service!"
}
```

**Response:**
```json
{
  "sentiment": "Positive",
  "confidence": 0.92,
  "probability_positive": 0.92,
  "probability_negative": 0.08,
  "cleaned_text": "restaurant amazing food great service",
  "word_contributions": [
    {"word": "amazing", "weight": 0.45},
    {"word": "great", "weight": 0.38}
  ]
}
```

### 2. Predict Batch Reviews
```http
POST /api/predict-batch
Content-Type: application/json

{
  "reviews": [
    "Great food and service!",
    "Worst experience ever",
    "Average restaurant"
  ]
}
```

**Response:**
```json
{
  "results": [
    {"review": "Great food and service!", "sentiment": "Positive", "confidence": 0.85},
    {"review": "Worst experience ever", "sentiment": "Negative", "confidence": 0.91},
    {"review": "Average restaurant", "sentiment": "Negative", "confidence": 0.52}
  ],
  "summary": {
    "total": 3,
    "positive": 1,
    "negative": 2,
    "pos_percentage": 33.33,
    "neg_percentage": 66.67
  }
}
```

### 3. Get Model Metrics
```http
GET /api/metrics
```

Returns accuracy, confusion matrix, classification report, top positive/negative words, and dataset statistics.

### 4. Get Reviews (Paginated)
```http
GET /api/reviews?page=1&limit=20&search=delicious&sentiment=positive
```

**Parameters:**
- `page`: Page number (default: 1)
- `limit`: Reviews per page (default: 20, max: 100)
- `search`: Search term (optional)
- `sentiment`: Filter by "positive", "negative", or "all" (default: all)

**Response:**
```json
{
  "reviews": [...],
  "total": 100,
  "page": 1,
  "limit": 20,
  "pages": 5
}
```

## Project Structure

```
restaurant sentiment analysis/
├── main.py                 # FastAPI application & API endpoints
├── train_model.py          # Model training script
├── Reviews.tsv             # Dataset (reviews with labels)
├── model.pkl               # Trained Logistic Regression model
├── vectorizer.pkl          # TF-IDF vectorizer
├── model_metrics.json      # Model performance metrics
├── restaurant_analysis.ipynb # Jupyter notebook (analysis)
├── README.md               # This file
├── static/
│   ├── index.html          # Frontend UI
│   ├── app.js              # Frontend JavaScript
│   └── style.css           # Frontend styling
└── venv/                   # Virtual environment
```

## Model Details

### Training Data
- **Dataset**: Reviews.tsv (restaurant reviews with sentiment labels)
- **Train/Test Split**: 70% training, 30% testing
- **Algorithm**: Logistic Regression
- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Max Features**: 3000
- **N-grams**: 1-2 (unigrams and bigrams)

### Preprocessing
1. Remove special characters
2. Convert to lowercase
3. Tokenization
4. Lemmatization (using NLTK WordNetLemmatizer)
5. Stop word removal

### Performance
- **Accuracy**: ~78%
- See `/api/metrics` endpoint for detailed metrics

## Development

### Hot Reload
The server runs with `--reload` flag, so changes to `main.py` will automatically reload the server.

### Adding New Features
1. Modify or add endpoints in `main.py`
2. Update `static/app.js` for frontend changes
3. The server will reload automatically

### Debugging
- Check console logs in browser (F12)
- Check terminal output for server logs
- Review API responses in Network tab

## Future Enhancements

- [ ] Add confidence score visualization
- [ ] Implement caching for frequently analyzed reviews
- [ ] Add data export functionality (CSV, JSON)
- [ ] Implement user authentication
- [ ] Add more NLP models (BERT, DistilBERT)
- [ ] Deploy to cloud (AWS, GCP, Azure)

## Troubleshooting

### Model Not Loading
- Ensure `model.pkl` and `vectorizer.pkl` exist
- Run `python train_model.py` to regenerate them

### Port Already in Use
If port 8000 is busy, use a different port:
```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

### CORS Issues
The application has CORS enabled for development. If issues persist, check `main.py` middleware configuration.

### Mobile Connection Issues
- Ensure firewall allows port 8000
- Check that phone and PC are on the same network
- Use the correct PC IP address

## License

This project is open source and available under the MIT License.

## Author

Created as a sentiment analysis project for restaurant reviews.

## Support

For issues or questions, please create an issue in the repository.
