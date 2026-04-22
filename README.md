# Email Spam Detection MLOps Pipeline

A complete machine learning pipeline for email spam detection with modular architecture and MLOps best practices.

## Project Structure

```
DVC-MLops-S3/
├── src/
│   ├── data_ingestion.py      # Data loading and ingestion
│   ├── data_preprocessing.py  # Text cleaning and preprocessing
│   ├── feature_engineering.py # Feature extraction and engineering
│   ├── model_building.py      # Model training and comparison
│   └── model_evaluation.py    # Model evaluation and testing
├── main.py                    # Main pipeline script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Features

- **Modular Design**: Each ML pipeline stage is in a separate module
- **Multiple Models**: Compares Logistic Regression, Random Forest, Naive Bayes, and SVM
- **Feature Engineering**: TF-IDF vectorization + custom text features
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Model Persistence**: Saves best model and preprocessing objects

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:
```bash
python main.py
```

This will:
1. Load/create sample spam detection dataset
2. Preprocess and clean text data
3. Engineer features (TF-IDF + custom features)
4. Train and compare multiple models
5. Evaluate model performance
6. Test on new email samples

## Pipeline Stages

### 1. Data Ingestion (`data_ingestion.py`)
- Creates sample email dataset with spam/ham labels
- Handles data loading and basic statistics

### 2. Data Preprocessing (`data_preprocessing.py`)
- Text cleaning (lowercase, remove punctuation, numbers)
- Stop word removal
- Train/test split

### 3. Feature Engineering (`feature_engineering.py`)
- TF-IDF vectorization
- Custom text features (length, special characters, spam keywords)
- Feature scaling

### 4. Model Building (`model_building.py`)
- Trains multiple classifiers
- Compares model performance
- Saves best performing model

### 5. Model Evaluation (`model_evaluation.py`)
- Comprehensive metrics calculation
- Performance visualization
- Testing on new samples

## Model Performance

The pipeline trains and compares:
- **Logistic Regression**: Linear classifier with regularization
- **Random Forest**: Ensemble of decision trees
- **Naive Bayes**: Probabilistic classifier (good for text)
- **SVM**: Support Vector Machine with RBF kernel

## Output Files

- `models/best_spam_detector_model.pkl`: Best performing model
- `models/tfidf_vectorizer.pkl`: TF-IDF vectorizer
- `models/scaler.pkl`: Feature scaler
- `results/evaluation_results.pkl`: Detailed evaluation results
- `results/evaluation_summary.csv`: Performance summary

## Example Usage

```python
from src.model_building import ModelBuilding
from src.feature_engineering import FeatureEngineering

# Load saved model
model_builder = ModelBuilding()
model = model_builder.load_model('best_spam_detector')

# Load feature engineering objects
feature_engineer = FeatureEngineering()
feature_engineer.load_feature_engineering_objects()

# Predict on new email
new_email = "Congratulations! You won $1000!"
# ... (preprocessing and feature engineering steps)
prediction = model.predict(features)
```

## Next Steps

- Integrate with DVC for data versioning
- Add model monitoring and drift detection
- Implement A/B testing framework
- Deploy model as REST API
- Add more sophisticated NLP features