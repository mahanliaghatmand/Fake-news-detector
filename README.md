# 📰 Fake News Detector

A deep learning application that classifies news articles as **Real** or **Fake** based on their text content, using a Bidirectional LSTM neural network. The project includes a full training pipeline (Jupyter Notebook), a trained model, and an interactive Streamlit web app for live predictions.

---

## 📋 Overview

This project trains a text classification model on ~5,800 labeled news articles to detect fake news. Article text is tokenized, converted into padded sequences, and passed through an Embedding + Bidirectional LSTM network to output the probability that an article is genuine.

### 🔗 https://fake-news-detector-mahan-liaghatmand.streamlit.app/

A lightweight **Streamlit web app** is included so predictions can be made by simply pasting article text into a browser — no coding required.

> ⚠️ **Important:** An audit of the training data uncovered a significant leakage signal — see [Known Limitations](#-known-limitations) before using this model for anything beyond a demo. Full details are in the accompanying [project report](#-project-report).

---

## ✨ Features

- **Bidirectional LSTM classifier** built with TensorFlow/Keras
- **~99.7% test accuracy** (see caveats below)
- **Text preprocessing pipeline**: tokenization + sequence padding (vocab size 20,000, max length 300)
- **Interactive Streamlit UI** — paste any article and get an instant Real/Fake prediction with a confidence score
- **Pre-trained, ready-to-use model** — no retraining required
- **Full transparency report** documenting data quality issues, model architecture, and evaluation results

---

## 🗂️ Project Structure

```
├── news_dataset.csv                    # Training dataset (5,829 labeled articles)
├── News_Fake_Detection_ipynb.txt       # Model training & evaluation notebook (JSON/ipynb)
├── model.pkl                           # Trained Keras BiLSTM model (pickled)
├── Tokenzier.pkl                       # Fitted Keras Tokenizer (vocab = 20,000)
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
└── Fake_News_Detection_Report.pdf      # Full project report (approach, findings, results)
```

> **Note:** `model.pkl` is required to run the app but may not always be attached alongside this README depending on where the project files are shared — make sure it's present in the same folder as `app.py` before launching.

---

## 🧠 Model Details

| Layer | Output Shape | Parameters |
|---|---|---|
| Embedding (vocab=20,000, dim=128) | (None, 300, 128) | 2,560,000 |
| Bidirectional LSTM (64 units) | (None, 128) | 98,816 |
| Dropout (0.5) | (None, 128) | 0 |
| Dense (32, ReLU) | (None, 32) | 4,128 |
| Dropout (0.3) | (None, 32) | 0 |
| Dense (1, Sigmoid) | (None, 1) | 33 |

**Total parameters:** 2,662,977 · **Optimizer:** Adam · **Loss:** Binary Crossentropy · **Regularization:** Early stopping (patience = 3, best weights restored)

### Performance

| Metric | Train | Test |
|---|---|---|
| Accuracy | 100.0% | 99.66% |
| Precision | — | 99.50% |
| Recall | — | 99.83% |

*(Evaluated on a held-out 20% test split — 1,166 articles. Full confusion matrix and training curves are in the [project report](#-project-report).)*

---

## ⚠️ Known Limitations

The training data contains a strong **wire-service leakage signal**: 99.6% of articles labeled *Real* contain the word "Reuters," versus only 1.8% of articles labeled *Fake*. This means the model can achieve very high accuracy largely by detecting wire-service formatting (e.g. a "CITY (Reuters) -" dateline) rather than learning to distinguish misinformation from legitimate reporting on content or style alone.

**Before production use:**
- Strip wire-service datelines/agency names from text and retrain to get a realistic, leakage-free accuracy estimate.
- Validate on out-of-distribution articles (different publishers, more recent dates).
- Treat this as a writing-pattern/style classifier, not a fact-checker — it does not verify the truthfulness of claims.

See the full [project report](#-project-report) for the complete data-quality audit and recommendations.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Streamlit App

1. Make sure `model.pkl` and `Tokenzier.pkl` are in the same folder as `app.py` (or update the paths at the top of `app.py`).
2. Launch the app:

```bash
streamlit run app.py
```

3. Paste an article's text into the text box (or pick a built-in example).
4. Click **Analyze** to see the Real/Fake prediction and confidence score.

### Retraining the Model

Open `News_Fake_Detection_ipynb.txt` (rename to `.ipynb` if needed) in Jupyter Notebook or Google Colab to explore the preprocessing pipeline, model architecture, training process, and evaluation — or to retrain on new/cleaned data.

---

## 📄 Project Report

A full write-up — covering the train/test split approach, data-quality and leakage audit, model architecture, validation results (with confusion matrix), training curves, and recommendations — is available in **`Fake_News_Detection_Report.pdf`**.

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras** — model architecture & training (Embedding, Bidirectional LSTM)
- **scikit-learn** — train/test split, evaluation metrics
- **pandas / NumPy** — data manipulation
- **Matplotlib** — visualization (accuracy/loss curves)
- **Streamlit** — interactive web app

---

## ⚠️ Disclaimer

This tool is built for **educational and demonstrative purposes only**. Due to the data leakage issue described above, it should **not** be relied upon as an authoritative fake-news detector. Always verify news through trusted, independent sources.

---

## 📄 License

This project is open for educational use. Feel free to fork, modify, and build upon it.
