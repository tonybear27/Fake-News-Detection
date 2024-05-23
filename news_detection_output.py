import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# 載入已儲存的模型
loaded_model = joblib.load('svm_fake_news_model.pkl')

# 文字清理函數（與訓練時的清理步驟一致）
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

# 定義預測函數
def predict_news(news_text):
    # 預處理新聞內文
    processed_text = preprocess(news_text)
    # 進行預測
    prediction = loaded_model.predict([processed_text])[0]
    # 獲取預測信賴分數
    decision_function = loaded_model.decision_function([processed_text])
    confidence_score = decision_function[0] if prediction == 'fake' else -decision_function[0]
    # 返回真假判斷和信賴分數
    return prediction[0], confidence_score

# input
news_text = input("請輸入新聞的內容: ")
label, confidence = predict_news(news_text)
print(f"Prediction: {label}")
print(f"Confidence Score: {confidence}")