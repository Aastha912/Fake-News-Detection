# Fake News Detection using Machine Learning & Deep Learning

This project focuses on detecting **fake vs real news articles** using **Natural Language Processing (NLP)** techniques.  
Two models are implemented and compared:

- **TF-IDF + Logistic Regression (Machine Learning)**
- **LSTM (Deep Learning)**

The goal is to analyze text data, build reliable classifiers, and evaluate their performance using real-world examples.

---

## Dataset

- **Dataset Name:** WELFake Dataset  
- **Source:** Zenodo  
- **Link:** https://zenodo.org/records/4561253  

The dataset contains labeled news articles with:
- `text` – news content  
- `label` – 0 (Real), 1 (Fake)

---

## Technologies Used

### Programming & Libraries
- Python
- Pandas, NumPy
- Regex (Text Cleaning)
- Matplotlib, Seaborn (Visualization)

### Machine Learning
- Scikit-learn
  - TF-IDF Vectorizer
  - Logistic Regression
  - Evaluation Metrics

### Deep Learning
- TensorFlow / Keras
  - Tokenizer
  - Embedding Layer
  - LSTM Network

---

## Exploratory Data Analysis (EDA)

- Removed missing values
- Analyzed label distribution (Fake vs Real)
- Studied article length distribution
- Generated word clouds for fake and real news
- Identified most frequent words

---

## Text Preprocessing

- Converted text to lowercase
- Removed URLs
- Removed special characters and numbers
- Removed extra spaces
- Filtered articles based on length (50–1000 words)

---

## Models Implemented

### 1️)TF-IDF + Logistic Regression
- Converts text into numerical features
- Fast and effective baseline model
- Performs well on large datasets

### 2️)LSTM (Long Short-Term Memory)
- Deep learning model for sequential text data
- Captures contextual patterns in text
- Uses word embeddings and sequence learning

---

## Model Performance

| Model                        | Accuracy | Precision | Recall | F1-Score |
|-----------------------------|----------|-----------|--------|----------|
| TF-IDF + Logistic Regression | 94%      | 0.94      | 0.94   | 0.94     |
| LSTM                         | 91%      | 0.90      | 0.91   | 0.90     |

Logistic Regression performed slightly better while being faster and simpler.

---

## Visualizations Included

- Label Distribution (Fake vs Real)
- Article Length Distribution
- Word Clouds (Fake News & Real News)
- Top 20 Most Frequent Words
- TF-IDF Feature Importance
- Confusion Matrices (LR & LSTM)
- LSTM Training vs Validation Accuracy
- LSTM Training vs Validation Loss
- Model Accuracy Comparison Bar Chart

---

## Real-World Testing

The project supports:
- **Single news article prediction**
- **Multiple news article testing**

Predictions are made using:
- TF-IDF + Logistic Regression
- LSTM model

Outputs are displayed as:
- **Fake**
- **Real**

---

## How to Run the Project (Google Colab)

1. Upload `WELFake_Dataset.csv`
2. Run all preprocessing and training cells
3. Train ML and DL models
4. Test custom news articles using input prompts

---

## Key Learnings

- Natural Language Processing techniques
- Feature extraction using TF-IDF
- Traditional ML vs Deep Learning comparison
- Model evaluation using real-world data
- Data visualization and interpretation

---

## Future Improvements

- Use pre-trained embeddings (GloVe / Word2Vec)
- Hyperparameter tuning
- Use Transformer models (BERT)
- Deploy as a web application
- Add multilingual support

---


