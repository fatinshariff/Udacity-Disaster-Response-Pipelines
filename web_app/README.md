# Disaster Response Pipeline Project

link: https://disaster-response-23.herokuapp.com

## 1. Project Overview

In this project, I'll apply data engineering to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. Using this web app, an emergency worker can input a new message and get classification results in several categories. The web app also display visualizations of the data. 

## 2. Project Components

There are three main components of this project:

### 2.1 ETL

This is the pipeline for Extract, Transform and Load. 
The file *data/process_data.py* include all the cleaning code that:

- load messages and categories dataset
- clean and merge both dataset
- save in a SQLite database

### 2.2 Machine Learning Pipeline

File *disasterapp/train_classifier.py* contains machine learning pipeline that:

- load data from SQL database
- split the data into a training set and a test set
- using machine learning pipeline to create model
- The data is vectorised and tfidf is performed
- Classifier is defined
- Using GridSearch CV to find the best parameter
- Saving the model in pickle file named *classifier.pkl*

### 2.3 Flask Web App

1. Run the following commands in the web_app directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
    
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves

    `python disasterapp/train_classifier.py data/DisasterResponse.db disasterapp/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Screenshots

Below are a few screenshots of the web app.

**1. Messages predicted into categories:**

![Alt text](readme_pics/Screenshot-1.png?raw=true "Title")


**2. Distribution of the messages categories:**
![Alt text](readme_pics/Screenshot-2.png?raw=true "Title")








