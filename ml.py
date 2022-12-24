import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import tweepy
import config
import nltk
import mysql.connector
# import sentiment_mod as s
mydb = mysql.connector.connect(host="localhost", user="root", passwd="8625058204", database='twitdb')
mycursor = mydb.cursor()
client = tweepy.Client(bearer_token = config.BEARER_TOKEN)
query = "63NotOutForever -is:retweet"
response = client.search_recent_tweets(query=query, max_results = 10, expansions=["author_id"])
users = {u["id"]: u for u in response.includes['users']}
for tweet in response.data:
    if users[tweet.author_id]:
        user = users[tweet.author_id]
        print(tweet.author_id)
        a = tweet.author_id
        print(user.username)
        b = user.username
        print(tweet.id)
        c = tweet.id
        print(tweet)
        print("\n\n\n\n")
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
df = pd.read_csv('news2.csv')
print(df.shape)
df.head()
print("No of missing label\t:", df[df['label'].isna()].shape[0])
print("No of missing text\t:", df[df['text'].isna()].shape[0])
print("No of missing id\t:", df[df['id'].isna()].shape[0])
print("No of missing title\t:", df[df['title'].isna()].shape[0])
df = df.fillna('')
df['title_text'] = df['title'] + '' + df['text']
df.head()
df = df[df['label']!='']
print(df['label'].unique())
df.loc[df['label'] == 'fake', 'label'] = 0
df.loc[df['label'] == 'Fake', 'label'] = 1
no_of_fakes = df.loc[df['label'] == 0].count()[0]
no_of_trues = df.loc[df['label'] == 1].count()[0]
print(no_of_fakes)
print(no_of_trues)
stop_words = set(stopwords.words('english'))
def clean(text):
   
    text = text.lower()
    text = re.sub(r'<[^>]*>', '', text) #html tags
    text = re.sub(r'@[A-Za-z0-9]+','',text)#username
    text = re.sub('https?://[A-Za-z0-9]','',text)#links
    # text = re.sub('[^a-zA-Z]',' ',text) #remove numbers
    word_tokens = word_tokenize(text)
    filtered_sentence = []
    for word_token in word_tokens:
        if word_token not in stop_words:
            filtered_sentence.append(word_token)

    text = (' '.join(filtered_sentence))
    return text

df['title_text'] = df['title_text'].apply(clean)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['title_text'].values)
X = X.toarray()
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.28, random_state=11)
pre = MultinomialNB()
pre.fit(X_train, y_train)
print(pre.score(X_train, y_train)*100) 
print(pre.score(X_test, y_test)*100)
predictions = pre.predict(X_test)
cm = confusion_matrix(y_test, predictions)
sentence = clean(ans)
vectorized_sentence = vectorizer.transform([sentence]).toarray()
print(pre.predict(vectorized_sentence))