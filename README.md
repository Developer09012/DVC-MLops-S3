# Email Spam Detection MLOps Pipeline with DVC

A complete machine learning pipeline for email spam detection with DVC for data versioning and pipeline orchestration.

## Project Structure

```
DVC-MLops-S3/
├── src/
│   ├── data_ingestion.py      # Data loading and ingestion
│   ├── data_preprocessing.py  # Text cleaning and preprocessing
│   ├── feature_engineering.py # Feature extraction and engineering
│   ├── model_building.py      # Model training
│   └── model_evaluation.py    # Model evaluation and testing
├── data/                      # Data directories (generated)
│   ├── raw/                   # Raw data
│   ├── interim/              # Intermediate processed data
│   └── processed/            # Final processed data
├── models/                    # Trained models
├── reports/                  # Evaluation metrics
├── logs/                     # Log files
├── param.yaml                # Pipeline parameters
├── dvc.yaml                  # DVC pipeline definition
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Features

- **DVC Pipeline**: Version-controlled ML pipeline with dependency tracking
- **Modular Design**: Each ML pipeline stage is in a separate module
- **Random Forest Model**: Trains Random Forest classifier for spam detection
- **Feature Engineering**: TF-IDF vectorization
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, ROC-AUC
- **DVC Live Integration**: Experiment tracking with dvclive
- **Parameter Management**: YAML-based parameter configuration

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install DVC (if not already installed):
```bash
pip install dvc
```

## Usage

### Run Complete Pipeline with DVC:
```bash
dvc repro
```

This will execute the complete pipeline:
1. **Data Ingestion**: Load spam.csv from GitHub, split into train/test
2. **Data Preprocessing**: Text cleaning, stopword removal, stemming
3. **Feature Engineering**: TF-IDF vectorization
4. **Model Building**: Train Random Forest classifier
5. **Model Evaluation**: Evaluate model and save metrics

### Run Individual Stages:
```bash
python src/data_ingestion.py
python src/data_preprocessing.py
python src/feature_engineering.py
python src/model_building.py
python src/model_evaluation.py
```

## Pipeline Configuration

### Parameters (`param.yaml`)
```yaml
data_ingestion:
  test_size: 0.21

feature_engineering:
  max_features: 40

model_building:
  n_estimators: 21
  random_state: 2
```

### DVC Pipeline (`dvc.yaml`)
Defines the 5-stage pipeline with dependencies:
- `data_ingestion` → `data_preprocessing` → `feature_engineering` → `model_building` → `model_evaluation`

## Data Flow

1. **Input**: Spam dataset from GitHub (`https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv`)
2. **Raw Data**: `data/raw/train.csv`, `data/raw/test.csv`
3. **Processed Data**: `data/interim/train_processed.csv`, `data/interim/test_processed.csv`
4. **Features**: `data/processed/train_tfidf.csv`, `data/processed/test_tfidf.csv`
5. **Model**: `models/model.pkl`
6. **Metrics**: `reports/metrics.json`

## Model Performance

The pipeline trains a **Random Forest Classifier** with:
- TF-IDF features for text representation
- Configurable number of estimators and random state
- Comprehensive evaluation metrics

## Output Files

- `models/model.pkl`: Trained Random Forest model
- `reports/metrics.json`: Evaluation metrics (accuracy, precision, recall, AUC)
- `dvclive/`: Experiment tracking with DVC Live
- `logs/`: Log files for each pipeline stage

## DVC Commands

```bash
# Run complete pipeline
dvc repro

# Show pipeline status
dvc status

# Show pipeline visualization
dvc dag

# Track new data
dvc add data/raw

# Push data to remote storage
dvc push

# Pull data from remote storage
dvc pull
```

## Example Usage

```python
import pickle
import pandas as pd

# Load trained model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load test features
test_data = pd.read_csv('data/processed/test_tfidf.csv')
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Make predictions
predictions = model.predict(X_test)
```

## Next Steps

- Add hyperparameter tuning with DVC experiments
- Integrate with cloud storage (S3, GCS)
- Add model monitoring and drift detection
- Implement CI/CD for ML pipeline
- Deploy model as REST API
- Add more sophisticated NLP features (BERT, word embeddings)