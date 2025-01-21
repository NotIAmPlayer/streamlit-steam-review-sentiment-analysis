import streamlit as st
import joblib
from scipy.sparse import hstack

# Load the saved models
vectorizer = joblib.load('bow_vectorizer.pkl')
scaler = joblib.load('scaler.pkl')
svm_model = joblib.load('svm_model.pkl')

# Title and description
st.title("Steam Game Review Sentiment Analyzer")
st.write("Enter review text and numerical data to check if the game is recommended.")

# User inputs
review_text = st.text_area("Enter the review text:")
hours_played = st.number_input("Enter hours played (numeric):", min_value=0.0, step=1.0)
helpfulness = st.number_input("Enter number of users who found it helpful (numeric):", min_value=0, step=1)
funniness = st.number_input("Enter number of users who found it funny (numeric):", min_value=0, step=1)

# Predict button
if st.button("Predict Recommendation"):
    if review_text.strip() == "":
        st.write("Please enter the review text.")
    else:
        # Preprocess the inputs
        bow_features = vectorizer.transform([review_text])
        numerical_features = scaler.transform([[hours_played, helpfulness, funniness]])
        combined_features = hstack([bow_features, numerical_features])
        
        # Make prediction
        prediction = svm_model.predict(combined_features)
        
        # Display result
        if prediction[0] == 1:
            st.success("The game is recommended!")
        else:
            st.error("The game is not recommended.")