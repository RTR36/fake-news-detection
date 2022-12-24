import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import tweepy
import pickle
import config
import nltk
import mysql.connector
from flask import Flask, render_template, request
app = Flask(__name__)
# import sentiment_mod as s
mydb = mysql.connector.connect(host="localhost", user="root", passwd="8625058204", database='twitdb')
mycursor = mydb.cursor()
client = tweepy.Client(bearer_token = config.BEARER_TOKEN)
query = "Zombievirus -is:retweet"
response = client.search_recent_tweets(query=query, max_results = 10, expansions=["author_id"])
users = {u["id"]: u for u in response.includes['users']}
for tweet in response.data:
    if users[tweet.author_id]:
        user = users[tweet.author_id]
        # print(tweet.author_id)
        a = tweet.author_id
        # print(user.username)
        b = user.username
        # print(tweet.id)
        c = tweet.id
        # print(tweet)
        # print("\n\n\n\n")
        d = str(tweet)
        sqlform = "insert into twitapi (userid, username, tweetid, tweet) values (%s,%s,%s,%s)"
        sql="select tweet from twitapi where id=4"
        data = [(a, b, c, d)]
        mycursor.executemany(sqlform, data)
        mycursor.execute(sql)
        myresult = mycursor.fetchall()
        for i in myresult:
            ans=i
            break
        mydb.commit()
def convertTuple(tup):
    str = ''
    for item in tup:
        str = str + item
    return str
tuplea = ans
str2 = convertTuple(tuplea)
# print(str2)
# print(type(str2))
df = pd.read_csv('data1.csv')
print(df.shape)
# df.head()
print("No of missing label\t:", df[df['Label'].isna()].shape[0])
print("No of missing text\t:", df[df['text'].isna()].shape[0])
# print("No of missing id\t:", df[df['id'].isna()].shape[0])
print("No of missing title\t:", df[df['title'].isna()].shape[0])
df = df.fillna('')
df['title'] = df['title']
# df.head()
df = df[df['Label']!='']
print(df['Label'].unique())
df.loc[df['Label'] == 'fake', 'Label'] = 0
df.loc[df['Label'] == 'Fake', 'Label'] = 0
no_of_fakes = df.loc[df['Label'] == 0].count()[0]
no_of_trues = df.loc[df['Label'] == 1].count()[0]
print(no_of_fakes)
print(no_of_trues)
stop_words = set(stopwords.words('english'))
port_stem=PorterStemmer()
def clean(text):
    text = text.lower()
    text = re.sub(r'<[^>]*>', '', text) #html tags
    text = re.sub(r'@[A-Za-z0-9]+','',text)#username
    text = re.sub('https?://[A-Za-z0-9]','',text)#links
    text = re.sub('[^a-zA-Z]',' ',text) #remove numbers
    word_tokens = word_tokenize(text)
    filtered_sentence = []
    for word_token in word_tokens:
        if word_token not in stop_words:
            filtered_sentence.append(word_token)
    text = (' '.join(filtered_sentence))
    return text
def new_det(message):
    print("-----------INSIDE new_det-----------------")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['title'].values)
    X = X.toarray()
    # print("Length of X.toarray : ",X.shape)
    y = df['Label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=11)
    #pre = MultinomialNB()
    filename="fake_new_model"
    # pickle.dump(pre,open(filename,'wb'))
    loaded_model=pickle.load(open(filename,'rb'))
    print("enter")
    loaded_model.fit(X_train, y_train)
    print("exit")
    print(loaded_model.score(X_train, y_train)*100) 
    print(loaded_model.score(X_test, y_test)*100)
    predictions = loaded_model.predict(X_test)
    sentence = clean(str2)
    vectorized_sentence = vectorizer.transform([sentence]).toarray()
    ans2 = loaded_model.predict(vectorized_sentence)
    return ans2[0] 
@app.route('/')     
def home():
    return render_template('index.html', str=str2)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = new_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")
    
if __name__ == '__main__':
    app.run(debug=True)