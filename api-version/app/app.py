import streamlit as st
import requests
from pydantic import BaseModel

# Define API endpoint
API_ENDPOINT = "https://kubrabuzlu-sentimentandintentionanalysis.hf.space/analyze/"

# Define data model for API request
class Text(BaseModel):
    text: str

# Create Streamlit app
st.title("Text Analysis App")

# Get text from user
input_text = st.text_area("Enter your text here:")

if st.button("Analyze"):
    # Send request
    response = requests.post(API_ENDPOINT, json=Text(text=input_text).dict())

    # Check response
    if response.status_code == 200:
        result = response.json()
        st.write("Sentiment:", result["sentiment"])
        st.write("Intention:", result["intention"])
    else:
        st.error("An error occurred while analyzing the text.")
