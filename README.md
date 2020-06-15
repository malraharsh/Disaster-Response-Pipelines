# Disaster Response Pipeline Project

## Introduction
In this project, I have made an API in which we can send a message on the website, which will then be processessed on the server and will give the prediction of label on which orgainzation can help with this problem.

## Requirements
- nltk 3.3.0
- numpy 1.15.2
- pandas 0.23.4
- scikit-learn 0.20.0
- sqlalchemy 1.2.12

## Motivation 
The motivation behind this project was for welfare of mankind. As during disasters and other calamities, people suffer and thus are in need of various everday items. So different organizations work to fulfil each requirement. But there are thousands of requests and it becomes difficult to identify the main objective of the request.

## Result
Thus this project was to showcase our data scientist skills to make and end-to-end machine learning pipeline which can be later on deployed on the web, which can automatically classify the request based on the messages, which can foster the help provided by organisations.


### Instructions:
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
