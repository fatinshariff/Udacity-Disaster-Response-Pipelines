# import libraries

import sys
import pandas as pd
import numpy as np
import nltk
import pickle

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import sent_tokenize
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from custom_transformer import StartingVerbExtractor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier

nltk.download(['punkt', 'wordnet'])

# load data from database
def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', con= engine)

    X = df.message
    y = df.iloc[:, 4:]
    col_names = y.columns
    return X, y, col_names


def tokenize(text):
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    #iterate through each token
    clean_tokens = []
    for tok in tokens:

        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok.lower().strip())
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
    ('features', FeatureUnion([

        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),

        ('starting_verb', StartingVerbExtractor())
    ])),

    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])
    
     # specify parameters for grid search
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.75, 1.0),
        'features__text_pipeline__tfidf__use_idf': (True, False)

    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs= 4, verbose = 2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    #predict on test data
    y_pred = model.predict(X_test)
    y_pred_pd = pd.DataFrame(y_pred, columns = category_names)

    print("\nBest Parameters:", model.best_params_)

    for column in category_names:
        print('--------------------------------------------------------\n')
        print(str(column))
        print(classification_report(Y_test[column], y_pred_pd[column]))


def save_model(model, model_filepath):
    with open(model_filepath , 'wb') as file:
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
