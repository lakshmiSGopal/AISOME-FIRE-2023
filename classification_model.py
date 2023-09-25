#importing necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
nltk.download('punkt')
nltk.download('stopwords')


def read_data(data): #funtion to display description of the data
    print("Number of rows in data: ", data.shape[0])
    print("Number of columns in data: ", data.shape[1])

def clean_data(data): #function to perform basic data pre-processing of tweets
    x = data.replace("\n"," ")
    data = data.lower()
    data = re.sub(r"(@[A-Za-z0-9_]+)|[^\w\s]|#|http\s+", " ", data)
    sw = stopwords.words('english')
    sw.remove('not')
    tweets_to_token = data
    tweets_to_token = word_tokenize(tweets_to_token)
    tweets_to_token = [word for word in tweets_to_token if not word in sw]
    text = " "
    text = text.join(tweets_to_token)
    return text

def train_model(data): #function performs data vectorization and splits the data to train and test data
    X = data['clean_tweet']
    y = data[['ingredients', 'side-effect', 'mandatory', 'rushed', 'ineffective', 'political', 'none', 'conspiracy',
              'country', 'pharma', 'unnecessary', 'religious']]
    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    Xfeatures = tfidf.fit_transform(X).toarray()
    X_train, X_test, y_train, y_test = train_test_split(Xfeatures, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def get_score(y_test, predictions): #function to compute the score of models
    print("Accuracy = ", accuracy_score(y_test, predictions))
    print("precision = ", precision_score(y_test, predictions, average='micro'))
    print("recall = ", recall_score(y_test, predictions, average='micro'))
    print("f1 = ", f1_score(y_test, predictions, average='micro'))


def svm_classifierchain(X_train, X_test, y_train, y_test): #SVM model wrapped in a classifier chain
    classifiersvc = ClassifierChain(LinearSVC(C=1, dual=False, penalty='l1', max_iter=5000))
    classifiersvc.fit(X_train, y_train)
    predictions = classifiersvc.predict(X_test)
    print("SVM wrapped in a classifier chain: ")
    get_score(y_test, predictions)

def lr_classifierchain(X_train, X_test, y_train, y_test): #LR model wrapped in a classifier chain
    classifierrf = ClassifierChain(LogisticRegression(C=10))
    classifierrf.fit(X_train, y_train)
    predictions = classifierrf.predict(X_test)
    print("Logistic Regression wrapped in a classifier chain: ")
    get_score(y_test, predictions)

def multioutput_svm(X_train, X_test, y_train, y_test): #SVM model wrapped in a multioutputclassifier
    svm = LinearSVC()
    clf = MultiOutputClassifier(svm, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("SVM wrapped in Multi output classifier: ")
    get_score(y_test, pred)

#common computations and function calls
data = pd.read_excel(r"aisome_ohe.xlsx") #reading the input data
read_data(data)
clean_se = []
for d in data['tweet']:
    clean = clean_data(d)
    clean_se.append(clean)
data['clean_tweet'] = clean_se
X_train, X_test, y_train, y_test = train_model(data)
svm_classifierchain(X_train, X_test, y_train, y_test)
lr_classifierchain(X_train, X_test, y_train, y_test)
multioutput_svm(X_train, X_test, y_train, y_test)
