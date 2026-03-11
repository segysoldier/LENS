# 🔬 LENS — NLP Suite

**LENS** is an intelligent text analysis platform built with Streamlit and scikit-learn, featuring four machine learning models for real-world NLP tasks.

---

## 🚀 Features

| Module | Description |
|---|---|
| 🛡️ **Spam Classifier** | Detects whether a message is spam or legitimate |
| 🌐 **Language Detection** | Identifies the language of any text (20+ languages) |
| 🍽️ **Food Review Sentiment** | Classifies food reviews as positive or negative |
| 📰 **News Classification** | Categorises news articles/headlines into topics |

Each module supports both **single text input** and **bulk CSV upload** with downloadable results.

---

## 📂 Project Structure

```
LENS/
├── LENS_UI.py            # Main Streamlit app (final UI)
├── lens_project.py       # Alternate UI version
├── nlp1.ipynb            # NLP experiments - Part 1
├── nlp2.ipynb            # NLP experiments - Part 2
├── nlp3.ipynb            # NLP experiments - Part 3
├── spam_classifier.pkl   # Trained spam detection model
├── lang_det.pkl          # Trained language detection model
├── news_cat.pkl          # Trained news categorisation model
├── review.pkl            # Trained sentiment analysis model
├── test_data/            # Sample test datasets
├── test_spam.csv
├── test_language.csv
├── test_news.csv
├── test_reviews.csv
└── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/segysoldier/LENS.git
cd LENS
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run LENS_UI.py
```

---

## 📊 Models Used

All models are trained using **scikit-learn** pipelines (TF-IDF + classifier):

- **Spam Classifier** — Naive Bayes / SVM on SMS/email data
- **Language Detector** — Multinomial NB on character n-grams
- **Sentiment Analyser** — Logistic Regression on food review data
- **News Categoriser** — Linear SVC on news article dataset

---

## 🧑‍💻 Team

Built with ❤️ by students exploring the world of **Natural Language Processing**.

- 📧 kaithwasyaman@gmail.com
- 📱 +91-7266091264

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
