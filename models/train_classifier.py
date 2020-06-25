import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

import re
import pickle
import nltk

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Loads data from the database
    
    Args:
        database_filepath: path to database
        
    Returns:
        (DataFrame) X: Features
        (DataFrame) Y: Labels
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('labeled_messages', engine)
    
    X = df['message']
    Y = df.drop(['id', 'original', 'message', 'genre'], axis=1)
    category_names = list(Y.columns)

    return X, Y, category_names


stop_words = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.WordNetLemmatizer()

def tokenize(text):
    """
    Tokenizes the text data
    
    Args:
    text str: Messages as text data
    
    Returns:
    words list: It first normalizing, tokenizing and lemmatizing, then process the data
    """
    text = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(i).strip().lower() for i in text if i not in stop_words]


def build_model():
    """
    Builds the model with GridSearchCV to find best combination of params
    
    Returns:
    Trained model after performing grid search on data
    """
    random_forest_clf = RandomForestClassifier(n_estimators=8)
    
    pipeline = Pipeline([('count', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(random_forest_clf))])
    
    parameters = {'clf__estimator__n_estimators':[7, 10, 20],
             'clf__estimator__max_depth':[None, 20],
             'clf__estimator__min_samples_leaf':[2, 4]}

    cv = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model against a test dataset to find the accuracy.
    
    Args:
        model: Trained model
        X_test: Test features
        Y_test: Test labels
        category_names: String array of category names
    """
    pred_y = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(col.upper())
        print(classification_report(Y_test[col], pred_y[:, i]))


def save_model(model, model_filepath):
    """
    Save the model to a Python pickle
    
    Args:
        model: Trained model
        model_filepath: Path where to save the model
    """
    import pickle
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()