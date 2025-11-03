
# 📧 Email Spam Detection App

A **Streamlit** web application that predicts whether an incoming email is **Spam** or **Ham** (Not-Spam) based on its text content. This project uses a **TF-IDF Vectorizer** combined with a **Multinomial Naive Bayes Classifier** to make real-time predictions.

---

## 🔗 Project Demo

You can run this app locally using Streamlit or deploy it on platforms like **Streamlit Cloud**.

---

## 🧰 Features

* **Interactive Streamlit UI** for text input and classification.
* Loads the `spam_ham_dataset (1).csv` file locally.
* **NLP Pipeline:** Uses `TfidfVectorizer` to convert email text into numerical features.
* **Multinomial Naive Bayes Classifier** for fast and effective text classification.
* Automatic training on application load.
* Displays **test accuracy** to show model performance.
* Provides **visual feedback** and **prediction probability** for the classified email.
* Simple data preprocessing to handle text cleaning (lowercase, stop word removal).

---

## 📁 Dataset

The project relies on the following local file:

`spam_ham_dataset (1).csv`

**Key Columns:**
* `text`: The raw content of the email (used as the feature).
* `label`: The target variable, indicating the outcome (`spam` or `ham`).

---

## ⚙️ Installation

1. Clone the repository:

git clone [https://github.com/YourUsername/Email-Spam-Detector.git](https://github.com/YourUsername/Email-Spam-Detector.git)
cd Email-Spam-Detector

Crucially, ensure you have the data file in the root directory:

spam_ham_dataset (1).csv


Install required packages:

pip install -r requirements.txt

requirements.txt content:

streamlit
pandas
numpy
scikit-learn


Run the app locally using:

streamlit run app.py



How It Works
Load Dataset The local CSV is loaded, filtering for the text and label columns.

Model Pipeline A Scikit-learn Pipeline is used, ensuring efficient preprocessing and training:

TF-IDF Vectorization: The TfidfVectorizer converts the raw email text into a matrix of TF-IDF features, which represent the importance of words in the document relative to the entire dataset.

Multinomial Naive Bayes: This classifier is trained on the TF-IDF features, making it highly effective for text classification tasks like spam detection.

Prediction UI

The user enters a sample email in the text area.

The text is passed through the trained pipeline.

The result is displayed with the predicted label (Spam or Ham) and the prediction probability.

💡 Future Improvements
Integrate NLTK/SpaCy for more advanced text cleaning (lemmatization, stemming).

Add a Word Cloud visualization for the most common spam and ham words.

Implement a Deep Learning model (e.g., RNN or BERT-based) for higher accuracy.

Deploy on Streamlit Cloud for public access.

Allow users to upload their own CSV dataset for training.
