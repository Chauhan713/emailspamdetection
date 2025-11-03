import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np

# -------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------
st.set_page_config(page_title="📧 Spam Email Detector", layout="wide")
st.title("📧 Email Spam Detection App")
st.info("Predict whether an email is 'Spam' or 'Ham' using a Machine Learning model.")

# Container for prediction results
prediction_output_container = st.empty()

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------
# Use a function to load and cache data for efficiency
@st.cache_data
def load_data(csv_file):
    """
    Loads data from the specified CSV file.
    Selects 'text' and 'label' columns and drops missing values.
    """
    try:
        df = pd.read_csv(csv_file)
        
        # Check for required columns
        if "text" in df.columns and "label" in df.columns:
            df = df[["text", "label"]]
            df.dropna(inplace=True) # Drop rows with missing text or label
            df['text'] = df['text'].astype(str) # Ensure text is string
            return df
        else:
            st.error("Error: CSV must contain 'text' and 'label' columns.")
            return None
    except FileNotFoundError:
        st.error(f"Error: The file '{csv_file}' was not found. Please make sure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Define the file path
csv_file = "spam_ham_dataset (1).csv"
df = load_data(csv_file)

# Only proceed if the dataframe is loaded successfully
if df is not None:

    # -------------------------------------------------------
    # Dataset Preview
    # -------------------------------------------------------
    with st.expander("📂 Preview Data", expanded=False):
        st.write(f"📊 Dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
        st.dataframe(df.head(10)) # Show first 10 rows

    # -------------------------------------------------------
    # Define Features (X) and Target (y)
    # -------------------------------------------------------
    X = df['text']
    y = df['label'] # Assuming 'label' column has 'spam' and 'ham'

    # -------------------------------------------------------
    # Train the Model (using a Pipeline)
    # -------------------------------------------------------
    try:
        # Split data to calculate accuracy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Create a pipeline that first vectorizes the text and then applies the classifier
        # Using max_features=5000 to keep the model lightweight
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000, lowercase=True)), # Vectorizer
            ('model', MultinomialNB(alpha=0.1)) # Classifier
        ])

        # Train the model on the full dataset for the best prediction
        # (We'll use the split data just for the accuracy score)
        
        # Train on the training split
        pipeline.fit(X_train, y_train)

        # Evaluate the model on the test split
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model trained successfully (Test Accuracy: **{acc:.2f}**)")

        # -------------------------------------------------------
        # User Input UI
        # -------------------------------------------------------
        with st.expander("🎯 Try Prediction (Input Email Text)", expanded=True):
            st.write("Enter the full text of an email below to classify it as Spam or Ham.")
            
            # Use a form to group the input and button
            with st.form(key="prediction_form"):
                user_input = st.text_area(
                    "Email Text", 
                    height=250, 
                    placeholder="Subject: You've won a prize!\n\nClick here to claim your $1,000,000..."
                )
                
                st.markdown("---")
                button_clicked = st.form_submit_button(
                    "🚀 Classify Email", 
                    type="primary", 
                    use_container_width=True
                )

        # -------------------------------------------------------
        # Prediction Output
        # -------------------------------------------------------
        if button_clicked:
            if user_input.strip():
                # Predict the class
                prediction = pipeline.predict([user_input])[0]
                # Get prediction probabilities
                probabilities = pipeline.predict_proba([user_input])[0]
                
                # Get the class names from the pipeline
                # This makes it robust even if classes are 0/1 or ham/spam
                class_names = pipeline.classes_
                
                # Create a dictionary of probabilities
                prob_dict = {label: prob for label, prob in zip(class_names, probabilities)}
                
                # Get the probability of the predicted class
                predicted_prob = prob_dict[prediction]

                with prediction_output_container.container():
                    st.subheader("📊 Prediction Results")
                    if prediction.lower() == "spam":
                        st.error(f"❌ **Predicted Outcome: Spam** ({predicted_prob:.1%} certainty)")
                    else:
                        st.success(f"✅ **Predicted Outcome: Ham** ({predicted_prob:.1%} certainty)")
                    
                    # Show detailed probabilities
                    st.write("All Probabilities:")
                    st.dataframe(pd.DataFrame(prob_dict, index=["Probability"]).T)

            else:
                # Show a warning in the output container if no text was entered
                with prediction_output_container.container():
                    st.warning("⚠️ Please enter some email text to classify.")

    except Exception as e:
        st.error("⚠️ Error during model training or prediction.")
        st.exception(e)

else:
    st.warning("Dataset could not be loaded. App cannot proceed. Please ensure 'spam_ham_dataset (1).csv' is available.")
