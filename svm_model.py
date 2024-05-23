import pandas as pd
import re
import nltk
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import joblib

# 下載NLTK的停用詞庫
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load data
def load_data(file_path, label):
    data = pd.read_csv(file_path)
    data['label'] = label
    return data

# Read data
train_fake = load_data('Dataset/training/Fake.csv', 'fake')
train_true = load_data('Dataset/training/True.csv', 'true')
val_fake = load_data('Dataset/validation/Fake.csv', 'fake')
val_true = load_data('Dataset/validation/True.csv', 'true')
test_fake = load_data('Dataset/testing/Fake.csv', 'fake')
test_true = load_data('Dataset/testing/True.csv', 'true')

# Combine data
train_data = pd.concat([train_fake, train_true], ignore_index=True)
val_data = pd.concat([val_fake, val_true], ignore_index=True)
test_data = pd.concat([test_fake, test_true], ignore_index=True)

# 文本清理函數
def clean(text):
    # 移除URL
    text = re.sub(r'http\S+', '', text)
    # 移除所有特殊字符
    text = re.sub(r'\W', ' ', text)
    # 移除單個字母
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # 移除多個空格
    text = re.sub(r'\s+', ' ', text)
    # 轉換為小寫
    text = text.lower()
    return text

# 移除停用詞
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# 清理並移除停用詞
train_data['text_clean'] = train_data['text'].apply(clean).apply(remove_stopwords)
val_data['text_clean'] = val_data['text'].apply(clean).apply(remove_stopwords)
test_data['text_clean'] = test_data['text'].apply(clean).apply(remove_stopwords)

# Define x and y
x_train = train_data['text_clean']
y_train = train_data['label']
x_val = val_data['text_clean']
y_val = val_data['label']
x_test = test_data['text_clean']
y_test = test_data['label']

# Create and optimize model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.5, min_df=2, ngram_range=(1,1))),
    ('clf', svm.SVC(kernel='linear', C=1.9, gamma='auto')),
])

parameters = {
    'tfidf__max_features': (3600, 10000)
}

f1_scorer = make_scorer(f1_score, average='micro', pos_label="fake")

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=10, verbose=1, scoring=f1_scorer)

print("Performing grid search...")
grid_search.fit(x_train, y_train)
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# Save the trained model
joblib.dump(grid_search, 'svm_fake_news_model.pkl')
print("模型以保存為 svm_fake_news_model.pkl")

# Evaluate on validation set
val_predictions = grid_search.predict(x_val)
print("Validation micro F1 score: " + str(f1_score(y_val, val_predictions, average='micro')))
print("Validation macro F1 score: " + str(f1_score(y_val, val_predictions, average='macro')))

# Final evaluation on test set
test_predictions = grid_search.predict(x_test)
print("Test micro F1 score: " + str(f1_score(y_test, test_predictions, average='micro')))
print("Test macro F1 score: " + str(f1_score(y_test, test_predictions, average='macro')))

# Plot confusion matrix
def make_confusion_matrix(cf, group_names=None, categories='auto', count=True, percent=True, cbar=True, xyticks=True,
                          xyplotlabels=True, sum_stats=True, figsize=None, cmap='Blues', title=None):
    blanks = ['' for i in range(cf.size)]
    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    if figsize is None:
        figsize = plt.rcParams.get('figure.figsize')

    if not xyticks:
        categories = False

    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if title:
        plt.title(title)
    plt.show()

labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
categories = ['Fake', 'Real']
svm_cf_matrix = confusion_matrix(y_test, test_predictions)
make_confusion_matrix(svm_cf_matrix, group_names=labels, categories=categories, cmap='Blues')
