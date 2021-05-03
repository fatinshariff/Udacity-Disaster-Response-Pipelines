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

from utils import tokenize
nltk.download(['punkt', 'wordnet'])

# load data from database
def load_data(database_filepath):
    """ Load data from specified database path.
    
    Args:
        database_filepath: Path to the database

    Output:
        X: Variables help to predict
        y: Target variables
        col_names: Column names of target variables

    """

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', con= engine)

    X = df.message
    y = df.iloc[:, 4:]
    col_names = y.columns
    return X, y, col_names


def build_model():
    """ 
    Pipeline to create a model. The data is vectorized and tfidf is performed first.
    Custom transformer is also performed.
    Classifier is defined.
    Parameters is set uand unsing GridSearch to find the best parameter.
    
    Output:
        cv: Final model that will be use to classify """
    
    pipeline = Pipeline([
    ('features', FeatureUnion([

        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),

        ('starting_verb', StartingVerbExtractor())
    ])),

    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
])
    
     # specify parameters for grid search
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.75, 1.0)
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs= 8, cv = 3, verbose = 2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluating model by their f1 score,precision and recall for each category.

    Args:
        model: Final model
        X_test: Test set for predicting
        Y_test: Target variable for test set 
        category_names: Target variable's names

    Output:
        F1 score, precision and recall for each category

    """

    #predict on test data
    y_pred = model.predict(X_test)
    y_pred_pd = pd.DataFrame(y_pred, columns = category_names)

    print("\nBest Parameters:", model.best_params_)

    for column in category_names:
        print('--------------------------------------------------------\n')
        print(str(column))
        print(classification_report(Y_test[column], y_pred_pd[column]))


def save_model(model, model_filepath):
    """
    Save model in a pickle file

    Args:
        model: Final model
        model_filepath: pickle file path

    """

    with open(model_filepath , 'wb') as file:
        pickle.dump(model, file)


def main():
    """
    Perform model building, training, predicting, evaluating and saving

    """

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
