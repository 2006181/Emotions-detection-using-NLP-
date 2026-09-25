# 😊 Emotion Detection using NLP

An NLP-based machine learning project that detects the emotion expressed in a given text.

The project takes a text sentence as input, performs preprocessing, converts the text into **Bag-of-Words (BOW)** features, and uses a trained machine learning model to predict the corresponding emotion.

## 🚀 Project Overview

The goal of this project is to build a simple text-based emotion detection system using Natural Language Processing and Machine Learning.

The project includes:

* NLP text preprocessing
* Bag-of-Words feature extraction
* Trained emotion classification model
* Emotion label mapping
* Streamlit web application

## 🧠 How It Works

```text
User Input
    ↓
Text Preprocessing
    ↓
Remove Punctuation
    ↓
Remove Numbers
    ↓
Bag-of-Words Vectorization
    ↓
Trained ML Model
    ↓
Emotion Prediction
    ↓
Display Emotion
```

## 🎯 Supported Emotions

The training dataset contains multiple emotion categories, including:

* 😊 Joy
* 😢 Sadness
* 😡 Anger
* ❤️ Love
* 😨 Fear
* 😲 Surprise

The dataset stores each sentence together with its corresponding emotion label.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLP
* Bag-of-Words
* Streamlit
* Pickle
* Jupyter Notebook

## 📂 Project Structure

```text
Emotions_detection-with-NLP/
│
├── final_project.ipynb
├── train.txt
│
├── app.py
│
├── emotion_model.pkl
├── bow_vectorizer.pkl
├── emotion_number.pkl
│
├── package.json
├── package-lock.json
├── .gitignore
└── README.md
```

### File Description

| File                  | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| `final_project.ipynb` | Main notebook containing the NLP and ML workflow              |
| `train.txt`           | Training dataset containing text and emotion labels           |
| `app.py`              | Streamlit application for emotion prediction                  |
| `emotion_model.pkl`   | Saved trained emotion classification model                    |
| `bow_vectorizer.pkl`  | Saved Bag-of-Words vectorizer                                 |
| `emotion_number.pkl`  | Mapping used to convert predicted numbers into emotion labels |
| `.gitignore`          | Files and folders excluded from Git                           |

## 🚀 Live Demo

[Streamlit](https://2006181-emotions-detection-using-NLP--app-4cln8n.streamlit.app/)

## 🔄 Text Preprocessing

Before prediction, the input text goes through preprocessing.

### 1. Convert to Lowercase

The input text is converted to lowercase.

```python
cleaned_text = text.lower()
```

### 2. Remove Punctuation

Punctuation and unnecessary characters are removed using a custom Python function.

```python
def remove_punctuation(text):
    new = ""

    for i in text:
        if i.isalnum() or i.isspace():
            new = new + i

    return new
```

### 3. Remove Numbers

Numbers are removed from the text before prediction.

```python
def remove_numbers(text):
    new = ""

    for i in text:
        if not i.isdigit():
            new = new + i

    return new
```

These preprocessing steps are also implemented in the Streamlit application.

## 🔤 Bag-of-Words

After preprocessing, the cleaned text is converted into numerical features using the saved **Bag-of-Words vectorizer**.

```python
text_bow = vectorizer.transform([cleaned_text])
```

The resulting numerical representation is then passed to the trained model for prediction.

## 🤖 Emotion Prediction

The trained model predicts a numerical emotion label.

```python
prediction = model.predict(text_bow)
```

The predicted number is then converted back into the corresponding emotion using the saved emotion mapping.

```python
emotion = list(emotion_number.keys())[prediction[0]]
```

Finally, the predicted emotion is displayed in the Streamlit application.

## 🖥️ Streamlit Application

The project includes a simple Streamlit interface where users can enter a sentence and click **Detect Emotion**.

Example:

```text
Input:
I am feeling very happy today!

Output:
Detected Emotion: joy
```

The application provides a text area for user input and a **Detect Emotion** button.

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/2006181/Emotions-detection-using-NLP-.git
```

### 2. Move into the project folder

```bash
cd Emotions_detection-with-NLP
```

### 3. Install Python dependencies

```bash
pip install pandas numpy scikit-learn streamlit nltk jupyter
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Dataset

The project uses `train.txt` as the training dataset.

Each record contains a text sentence and its associated emotion label separated by `;`.

Example:

```text
i am feeling grouchy;anger
i feel romantic too;love
i feel so sad and hopeless;sadness
```

The dataset contains natural-language sentences representing different emotional states.

## 📚 What I Learned

Through this project, I practiced:

* Natural Language Processing
* Text preprocessing
* Python string manipulation
* Emotion classification
* Bag-of-Words
* Machine Learning model training
* Model serialization using Pickle
* Loading trained models for prediction
* Building a Streamlit ML application
* Connecting an ML model with a user interface
* Using Git and GitHub for version control

## 👨‍💻 Author

**Rahul Gupta**

B.Tech CSE (AI Specialization)
ABES Institute of Technology, Ghaziabad

---

⭐ If you found this project interesting, consider giving the repository a star!
