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
     
        
        
 
