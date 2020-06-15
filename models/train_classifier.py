import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
import numpy as np



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
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('labeled_messages', engine)
    
    X = df['message']
    Y = df.drop(['id', 'original', 'message', 'genre'], axis=1)
    category_names = list(Y.columns)

    return X, Y, category_names


stop_words = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.WordNetLemmatizer()

def tokenize(text):
    text = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(i).strip().lower() for i in text if i not in stop_words]


def build_model():
    random_forest_clf = RandomForestClassifier(n_estimators=8)
    pipeline = Pipeline([('count', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(random_forest_clf))])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    pred_y = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(col.upper())
        print(classification_report(Y_test[col], pred_y[:, i]))


def save_model(model, model_filepath):
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