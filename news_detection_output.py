import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from contextlib import redirect_stdout
import nltk
import warnings
import os
from colorama import Fore, Style
import colorama
colorama.init(autoreset=True)

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.base')
nltk.download('stopwords', quiet=True)

# load the pretrained model
global language
language = input('Please enter the language first (French/English/Spanish): ')
flag = input('Please input title or text (title/text): ')

loaded_model = joblib.load(f'{language}_model_{flag}.pkl')
language = language.lower()

# clear the data
def clean(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words(language))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def preprocess(text):
    text = clean(text)
    text = remove_stopwords(text)
    return text

# Define the prediction model
def predict_news(news_text):
    # pre-handle the content
    processed_text = preprocess(news_text)
    
    # prediction 
    prediction = loaded_model.predict([processed_text])[0]
    
    # get the confidence score
    decision_function = loaded_model.decision_function([processed_text])
    confidence_score = decision_function[0] if prediction == 'fake' else -decision_function[0]
    
    return prediction[0], confidence_score

# input
news_text = input(f"Please enter the news {flag}: ")
label, confidence = predict_news(news_text)

if label == 't':
    label = 'Real News'
    print(f"{Fore.GREEN}Prediction: {label}")
else:
    label = 'Fake News'
    print(f"{Fore.RED}Prediction: {label}")

print(f"Confidence Score: {confidence}")