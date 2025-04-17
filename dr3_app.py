import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import openai
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load models
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# UI
st.title("DR3 Mood-based Recommendation System")
st.markdown("**Created by Narendar Reddy Rentala**")

if 'history' not in st.session_state:
    st.session_state['history'] = []

for user_input, bot_response in st.session_state['history']:
    st.write(f"User: {user_input}")
    st.write(f"Bot: {bot_response}")

user_input = st.text_input("How can I assist you today?")

if user_input:
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model(**inputs)
    mood = torch.argmax(outputs.logits, dim=1).item()

    if mood == 0:
        response = "It seems like you're feeling a bit down. How can I assist you?"
    elif mood == 1:
        response = "You're in a good mood! Let's talk about something fun!"
    else:
        response = "You seem neutral. Let's see how we can make today better."

    # GPT response
import openai
import streamlit as st

# Retrieve the API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Now you can call the OpenAI API as usual
response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Hello, how are you?",
    max_tokens=100
)

print(response.choices[0].text.strip())

    st.session_state['history'].append((user_input, response))
    st.write(f"Bot: {response}")
