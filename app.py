from flask import Flask,render_template,url_for,request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re
from os.path import join, dirname, realpath
import os
# load the model from disk
filename = 'Sentiment_analysis.pkl'
model = pickle.load(open(filename, 'rb'))
filename = 'transformer.pkl'
vectorizer = pickle.load(open(filename, 'rb'))
app = Flask(__name__)
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result',methods=['POST'])
def predict():
    if request.method == 'POST':
        uploaded_file = request.files['myfile']
        if uploaded_file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            # set the file path
            uploaded_file.save(file_path)


            message = pd.read_csv('static/files/test.csv')

            message=message.iloc[:,1]
            message = [cleantweet(i) for i in message]
            message = [remove_emojis(i) for i in message]
            message = [i.lower() for i in message]
            message = [i.strip() for i in message]
            message = vectorizer.transform(message)
            my_prediction = model.predict(message)

            return render_template('result.html',value=round(my_prediction.mean(),3))

def cleantweet(tweet):
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)  # to remove @
    tweet = re.sub(r'[0-9]', '', tweet)
    tweet = re.sub(r'#', '', tweet)  # to remove hashtags
    tweet = re.sub(r'RT[\s]', '', tweet)  # to remove retweets
    tweet = re.sub(r'https?:\/\/\S+', '', tweet)  # to remove hyperlinks
    tweet = re.sub(r'[^\w\s]', '', tweet)  # to remove punctuations
    tweet = re.sub(r'\n', '', tweet)  # to remove next line
    tweet = re.sub(r'_', '', tweet)  # to remove underscore
    tweet = re.sub(" \d+", "", tweet)  # to remove numericals
    return tweet

def remove_emojis(tweet):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', tweet)


if __name__ == '__main__':
	app.run(debug=True)


