# Disaster Response Pipeline Project

## Project Overview

Analyzing disaster data from Figure Eight to build a model for a web app that classify the disaster messages. In this web app, an emergency worker can input a new message and get classification results in several categories. The web app also display visualizations of the data. 

## File Description

- app
    - template
        - master.html :  main page of web app
        - go.html : classification result page of web app
    - custom_transformer.py : customised transformer (StartingVerbExtractor)
    - run.py : Flask file that runs app

- data
    - disaster_categories.csv : data to process 
    - disaster_messages.csv : data to process
    - process_data.py : read the dataset, clean the data, and then store it in a SQLite database
    - InsertDatabaseName.db  : database to save clean data to

- models
    - custom_transformer.py : customised transformer (StartingVerbExtractor)
    - train_classifier.py : split the data into a training set and a test set. Then, using machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV, a final model will be outputted. It uses the message column to predict classifications for 36 categories (multi-output classification). Finally, the model is exported to a pickle file
    - classifier.pkl  # saved model 



## How to run the web app :

1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Screenshots

Below are a few screenshots of the web app.

**1. Messages predicted into categories:**

![Alt text](readme_pics/Screenshot-1.png?raw=true "Title")


**2. Distribution of the messages categories:**
![Alt text](readme_pics/Screenshot-2.png?raw=true "Title")








