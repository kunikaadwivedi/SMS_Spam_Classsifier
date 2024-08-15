import streamlit as st
import pickle
import pandas as pd

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Function to classify text
def classify_text(text):
    transformed_text = tfidf.transform([text])
    prediction = model.predict(transformed_text)
    return 'Spam' if prediction == 0 else 'Ham'

# Streamlit app
st.markdown('<h1 class="custom-title">SMS SPAM CLASSIFIER</h1>', unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .custom-font {
        font-family: Georgia, sans-serif;
        font-size: 20px;
        color: rgb(0, 0, 255);
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("""
 <p class="custom-font">
Welcome to the SMS SPAM/HAM CLASSIFIER! Enter the SMS text to classify it as Spam or Ham
""",
    unsafe_allow_html=True)

# Input text box
input_text = st.text_area('Enter SMS text here')

# Button to classify
if st.button('Classify'):
    if input_text:
        result = classify_text(input_text)
        st.write(f'The message is: {result}')
    else:
        st.write('Please enter some text to classify.')

# Display example data
if st.checkbox('Show example data'):
    example_data = pd.DataFrame({
        'text': [
            'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.',
            'Nah I don?t think he goes to usf, he lives around here though.'
        ],
        'label': ['Spam', 'Ham']
    })
    st.write(example_data)
prompt = st.chat_input("Tell us your Experience")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")
st.markdown("""
---
*Created by Kunikaa Dwivedi with "love" using [Streamlit](https://streamlit.io/)*
""")
