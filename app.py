from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a default threshold value
threshold = 0.5

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    # Load user input
    edge_followed_by = float(request.form['edge_followed_by'])
    edge_follow = float(request.form['edge_follow'])
    username_length = int(request.form['username_length'])
    username_has_number = int(request.form['username_has_number'])
    full_name_has_number = int(request.form['full_name_has_number'])
    full_name_length = int(request.form['full_name_length'])
    is_private = int(request.form['is_private'])
    is_joined_recently = int(request.form['is_joined_recently'])
    has_channel = int(request.form['has_channel'])
    is_business_account = int(request.form['is_business_account'])
    has_guides = int(request.form['has_guides'])
    has_external_url = int(request.form['has_external_url'])

    # Make prediction
    user_input = [[edge_followed_by, edge_follow, username_length,
                   username_has_number, full_name_has_number, full_name_length,
                   is_private, is_joined_recently, has_channel, is_business_account, has_guides, has_external_url]]

    prediction_proba = 0
    for clf in model.estimators_:
        prediction_proba += clf.predict_proba(user_input)[0][1]
    
    prediction_proba /= len(model.estimators_)
    
    # Determine prediction label
    prediction = "REAL" if prediction_proba >= threshold else "FAKE"

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
