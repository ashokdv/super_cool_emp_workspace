import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
# import cPickle
import flask
from flask import Flask
import json
from flask import jsonify, request

app=Flask(__name__)
@app.route('/mood-predict',methods=['POST'])
def mood():
    content = request.get_json()
    features = pd.read_csv('moodData.csv')
    features.head(5)
    print('The shape of our features is:', features.shape)
    # print(features.describe())
    # One-hot encode the data using pandas get_dummies
    features = pd.get_dummies(features)
    # Display the first 5 rows of the last 12 columns
    print(features)
    features.iloc[:,:].head(5)
    import numpy as np
    labels = np.array(features['happinessLevel'])
    features= features.drop('happinessLevel', axis = 1)
    # Saving feature names for later use
    feature_list = list(features.columns)

    # Convert to numpy array
    features = np.array(features)
    # from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 42)


    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # Import the model we are using
    # from sklearn.ensemble import RandomForestRegressor
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels)

    # save the model to disk
    filename = 'finalized_model.pkl'
    # pickle.dump(rf, open(filename, 'wb'))


    with open(filename, 'wb') as f:
        pickle.dump(rf, f)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    print(test_features)
    print(predictions)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


    with open(filename, 'rb') as f:
        rf = pickle.load(f)
    print('Pickle file loaded')
    ran = int(content["ran"])
    year = int(content["year"])
    month=int(content["month"])
    day = int(content["day"])
    Week_Fri = int(content["Week_Fri"])
    Week_Mon = int(content["Week_Mon"])
    Week_Sat = int(content["Week_Sat"])
    Week_Sun = int(content["Week_Sun"])
    Week_Thu = int(content["Week_Thu"])
    Week_Tue = int(content["Week_Tue"])
    Week_Wed = int(content["Week_Wed"])

    test_features = [[ ran,year,month,day,Week_Fri ,Week_Mon ,Week_Sat ,Week_Sun ,Week_Thu ,Week_Tue ,Week_Wed]]
    predictions = rf.predict(test_features)
    print(predictions[0])
    x = predictions[0]
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    return jsonify(mood_predict=x)


if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)

