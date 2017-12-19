
#Authors: Max Miranda & Stackoverflow Users MrPromethee & Mandrek & Github User: yanofsky
import tweepy
from textblob import TextBlob
import csv
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
tb = Blobber(analyzer=NaiveBayesAnalyzer())

consumer_key = 'PsDN5seMkon3xK1gR6btxg7e8'
consumer_secret = 'wujHFh5vpnuPp8KqLY4yK38tYD1fxGqnf59rR9qIKqfAS5n97u'

access_key = '867029637351845889-dagRg9LaD5nTnCW64urUjpFfEC3qJUF'
access_secret = 'RpIRoA55RLVT9jqs2XgQPhPkAm4hCDIIfjyYsqVojZEaU'

def get_all_tweets_and_sentiments(screen_name):
	#Twitter only allows access to a users most recent 3240 tweets with this method

	#authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

	#initialize a list to hold all the tweepy Tweets
    alltweets = []

	#make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)

	#save most recent tweets
    alltweets.extend(new_tweets)

	#save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

	#keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print "getting tweets before %s" % (oldest)

		#all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)

		#save most recent tweets
        alltweets.extend(new_tweets)

		#update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print "...%s tweets downloaded so far" % (len(alltweets))

	#transform the tweepy tweets into a 2D array that will populate the csv
    #outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]

    #write the csv
    with open('%s_tweets_sentiments.csv' % screen_name, 'wb') as scorefile:
        scoreFileWriter = csv.writer(scorefile)
        for tweet in alltweets:
            analysis = tb(tweet.text)
            sent = analysis.sentiment
            if sent.p_pos > .52:
                opinion = "Positive"
            elif sent.p_neg > .52:
                opinion = "Negative"
            else:
                opinion = "Neutral"
            scoreFileWriter.writerow([tweet.text.encode("utf-8"), opinion])
	pass


if __name__ == '__main__':
	#pass in the username of the account you want to download
	get_all_tweets_and_sentiments("coindesk")
