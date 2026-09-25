import streamlit as st
import pickle


# -----------------------------
# Load Model
# -----------------------------
with open("emotion_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load BOW Vectorizer
with open("bow_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Load Emotion Mapping
with open("emotion_number.pkl", "rb") as file:
    emotion_number = pickle.load(file)


# -----------------------------
# Preprocessing Functions
# -----------------------------
def remove_punctuation(text):
    new = ""

    for i in text:
        if i.isalnum() or i.isspace():
            new = new + i

    return new


def remove_numbers(text):
    new = ""

    for i in text:
        if not i.isdigit():
            new = new + i

    return new


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("😊 Emotion Detection using NLP")

st.write(
    "Enter a sentence below and the model will predict the emotion."
)

text = st.text_area(
    "Enter your text:",
    placeholder="Example: I am feeling very happy today!"
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Detect Emotion"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        # Preprocessing
        cleaned_text = text.lower()

        cleaned_text = remove_punctuation(cleaned_text)

        cleaned_text = remove_numbers(cleaned_text)

        # Convert text into BOW
        text_bow = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(text_bow)

        # Convert number into emotion
        emotion = list(emotion_number.keys())[prediction[0]]

        # Display result
        st.success(f"Detected Emotion: {emotion}")