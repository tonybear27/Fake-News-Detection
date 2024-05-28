import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# load the pretrained model
loaded_model = joblib.load('svm_fake_news_model.pkl')

# clear the data
def clean(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
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
news_text = input("Please enter the news content: ")
label, confidence = predict_news(news_text)

print(f"Prediction: {label}")
print(f"Confidence Score: {confidence}")