import streamlit as st
import pickle

# Load the trained model
with open("trained_model.sav", "rb") as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open("vectorizer.sav", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet below to predict its sentiment.")

# Get user input
tweet = st.text_area("Tweet Text")

if st.button("Analyze Sentiment"):
    if tweet:
        # Convert text into numerical form using TF-IDF
        tweet_vectorized = vectorizer.transform([tweet])

        # Predict sentiment
        prediction = model.predict(tweet_vectorized)[0]

        # Display result
        if prediction == 1:
            st.success("Positive Sentiment 😊")
        else:
            st.error("Negative Sentiment 😞")
    else:
        st.warning("Please enter a tweet to analyze.")
