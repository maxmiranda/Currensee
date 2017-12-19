
# coding: utf-8

# # <font color='blue'>Data Science </font> & <font color = 'gold'> The Blockchain </font>
# 
# **Authors: Max Miranda, Jai Yarlagadda**             
# 
# **Credits to: Github User: yanofsky, Stackoverflow Users: MrPromethee & Mandrek**
# 
# The goal of this project was to take the last 3240 tweets (that's the number twitter's API allows for) from Coindesk's Twitter Account, and to apply natural language processing and data science tools to analyze what these tweets for sentiments towards cryptocurrencies. 
# 
# Tools: 
# *  *Tweepy* : in order to find the tweets 
# *  *TextBlob* : for sentiment analysis) 
# *  *Pandas* : data structure, all the table manipulation
# *  *Matplotlib* : draws pretty graphs
# *  *Numpy & SciPy* : sets of pre-written math functions

# In[220]:


################### For Parts 1-7 ####################
from datascience import *
import numpy as np
import scipy 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')

######## For Prediction Techniques (8-10) ############
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import math
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

from client.api.notebook import Notebook
ok = Notebook('exploration.ok')
_ = ok.auth(inline=True)


# ## 1. A tale of two tables:
# 
# The first step was to gain access to the Twitter API, which can be done really simply on apps.twitter.com. Which, as it turns out, is incredibly simple:
# 
# ![Screen%20Shot%202017-11-28%20at%2012.42.33%20PM.png](attachment:Screen%20Shot%202017-11-28%20at%2012.42.33%20PM.png)
# 

# The next was to be able to conduct simple positive or negative sentiment analysis on the data, which we used a Github Gist on dumping tweets from a twitter account that we combined with information on a StackOverflow post on how to conduct sentiment analysis using TextBlobber. (Thanks yanofsky, MrPromthee, & Mandrek).
# 
# The result was the following code: __[tweet_sentiment_anal.py](https://gist.github.com/maxmiranda/e3ce6cb8becae7b98a0f1a7df9469c4a)__

# We used *that* to produce the `coindesk_tweets_sentiments.csv` table.

# In[124]:


coindesk_tweets = Table.read_table("coindesk_tweets_sentiments.csv", encoding = "latin1")
print(coindesk_tweets.where("sentiment", "Positive").num_rows)
print(coindesk_tweets.where("sentiment", "Negative").num_rows)
print(coindesk_tweets.where("sentiment", "Neutral").num_rows)

coindesk_tweets


# `coindesk_tweets` will serve as our source table, and will help us to create two tables. One that keeps track of the number of mentions (a quantitative variable), and one that keeps track of general sentiment (a categorical variable).
# 
# First, one that counts the amount of mentions for a list of cryptocurrencies. 

# In[175]:


cryptos = make_array("Bitcoin", "Ethereum", "Ripple", "Bitcoin Cash", "Bitconnect", "Dash", "Ethereum Classic", "Iota", "Litecoin", "Monero","Nem","Neo","Numeraire","Stratis","Waves" )

num_mentions = make_array()

for crypto in cryptos: 
    num_mentions_for_one_capitalized = coindesk_tweets.where('content', are.containing(crypto)).num_rows
    num_mentions_for_one_upper = coindesk_tweets.where('content', are.containing(crypto.upper())).num_rows
    num_mentions_for_one_lower = coindesk_tweets.where('content', are.containing(crypto.lower())).num_rows
    
    num_mentions_for_one = num_mentions_for_one_lower + num_mentions_for_one_upper + num_mentions_for_one_capitalized
    num_mentions = np.append(num_mentions, num_mentions_for_one)
cryptos_num_mentions = Table().with_columns("crypto", cryptos,
                                         "num of mentions", num_mentions)
cryptos_num_mentions.show(len(cryptos))


# Now, time for the crytocurriences vs. their coindesk sentiment. First, we will create a new table that has both the original coindesk tweets and lists any cryptos mentioned. 

# In[176]:


cryptos_for_tweets = []

for tweet in coindesk_tweets.column("content"): # creating new column that can be appended to coindesk_tweets to give easy access to which tweets reference which cryptos   
    cryptos_for_a_tweet = []
    for crypto in cryptos: 
        if tweet.rfind(crypto) != -1:
            cryptos_for_a_tweet.append(crypto)
    
    cryptos_for_tweets.append(cryptos_for_a_tweet)
        
cryptos_tweets = coindesk_tweets.with_column("crypto mentioned", cryptos_for_tweets)

cryptos_tweets


# Now, to finally create the second table we will subtract the number of positive tweets by the number of negative tweets, to gain a general sentiment.

# In[184]:


sentiments = make_array()

for crypto in cryptos: 
    positive_num = cryptos_tweets.where("crypto mentioned", are.containing(crypto)).where("sentiment", "Positive").num_rows
    negative_num = cryptos_tweets.where("crypto mentioned", are.containing(crypto)).where("sentiment", "Negative").num_rows
    if positive_num - negative_num > 0: 
        sentiment = "Positive"
    elif positive_num - negative_num < 0:
        sentiment = "Negative"
    else: 
        sentiment = "Neutral"
    sentiments = np.append(sentiments, sentiment)

cryptos_sentiments = Table().with_columns("crypto", cryptos, 
                                         "general sentiment", sentiments)

cryptos_sentiments.show(cryptos_sentiments.num_rows)


# ## 2-4. Visualizing the variables

# In[178]:


cryptos_num_mentions = cryptos_num_mentions.sort("num of mentions", descending = True)
cryptos_num_mentions.barh('crypto')


# Clearly, in recent history, Bitcoin is far and away the most reported on cryptocurrency by Coindesk.

# In order to do the same for the categorical variable, we'll first group the table. The result will be a table that shows the amount of cryptocurrencies that recieved negative, positive and neutral sentiments from Coindesk. 

# In[179]:


grouped = cryptos_sentiments.group("general sentiment") 

fig1, ax1 = plots.subplots() #all code is almost exactly copied from pyplot's documentation
labels = grouped.column("general sentiment")
explode = (.2, 0, 0)
sentiments = grouped.column("count")


ax1.pie(sentiments, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)

ax1.axis('equal')  

plots.show()


# As would make sense for a business based on cryptocurrencies, Coindesk has about five times as many  positive sentiments on cryptocurrencies than it does negative. It is important to take into account, that this is only a small, 15-cryptocurrency sample, the actual pie chart of the entire population would have a significantly larger neutral slice as there are hundreds of cryptocurrencies, and it is nearly impossible that Coindesk could get the opportunity to tweet on all of them. 

# ## 5. Joining tables
# 
# Now, let's add in the part everyone actually cares about, the price! We'll do so by creating a new table for the prices and then joining all of the existing tables. 
# 
# Note: prices from https://coinmarketcap.com/ on 11/28/2017 at approximately 3:47 P.M.

# In[202]:


#Creating new table for prices
prices = make_array(10084.40, 472.24, 0.301084, 1552.62, 285.63, 627.05, 34.02, 1.48, 95.53, 198.75, 0.249141, 38.38, 15.31, 6.41, 5.83) 
crypto_prices = Table().with_columns( "crypto", cryptos, 
                                    "price", prices)

#Joining all existing data
existing_data = cryptos_sentiments.join("crypto", cryptos_num_mentions)
crypto_data = existing_data.join("crypto", crypto_prices)

crypto_data


# It is important to note that one could not have simply used with_column on prices, as the order of the cryptocurrencies might have been different, since we sorted one of the tables before creating the bar graph. If the order were thrown off that would have horrible ramifications on our interpretation of the data.    

# ## 6. Hypothesizing
# 
# Creating a hypothesis and testing it is what data science is the value of data science! For the sake of simplicity, let's choose a basic hypothesis to be able to test.
# 
# **Null Hypothesis:** There is no relationship between the price of a cryptocurrency and the number of times that Coindesk will mention it. 
# 
# **Alternative Hypothesis:** The larger a cryptocurrency's price, the more mentions it will get from Coindesk. 
# 

# ## 7. Testing the hypothesis
# 
# In order to test this hypotheis, we will do the following:
# 1. <font color = 'blue'> Calculate the correlation of the relationship. </font> 
# 2. <font color = 'blue'>  Calculate the probability that this correlation would have occurred. </font>
# 3. <font color = 'blue'> We will use a <font color = 'gold'>p-value cutoff of 5% </font>, meaning that this <font size = '4'> **correlation is only significant if there is a less than 5% chance that this correlation would be this high.** </font> </font>
# 

# In[216]:


x = crypto_data.column("price")
y = crypto_data.column("num of mentions")
'''
def standard_units(nums):
    """Return an array where every value in nums is converted to standard units."""
    return (nums - np.mean(nums)) / np.std(nums) 

r = np.average(standard_units(x) * standard_units(y))
'''

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y) 
if p_value < .05:
    print("There is a correlation, " + str(r_value) + ", and it's statistically significant!")
else: 
    print("Looks like any tiny correlation here is statistically insignificant.")


# Yay!! We can reject the null hypothesis, there actually was a correlation where we though there was. Now, it's time for something a little more advanced.

# ## 8-10. Using Machine Learning to predict Bitcoin Price
# 
# We can begin by stating a major disclaimer about these predictions. As everyone knows, this is an incredibly volatile market, and the machine learning algorithm we're using through SciPy is a linear regression. However, as you can see in the sample below, a linear relationship might not be the best way to describe the changes in crypto prices over time. As well, the practice of predicting future prices based off of past prices is one that's been oft-used but never confirmed a scientific method of predicting performance.  
# <font color = 'white'> data:image/jpeg;base64,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![image.png](attachment:image.png)
# </font>

# The prediction technique we're applying is machine learning, using Scikit Learn. The reason one would choose this prediction technique is that it goes beyond a simple linear regression, rather it takes a set of variables, and uses an algorithm developed with these variables to predict the future. This is clearly superior to basic linear regression because basic linear regression only uses one value to predict. 

# In[307]:


bitcoin = Table.read_table("bitcoin.csv") #retrieved from Quandl: includes: Date, Open, High, Low, Close, Volume (BTC), Volume (Currency), Weighted Price 
bitcoin


# In[315]:


#creating a couple of factors like volatility and percent_daily_change in order to help the machine learning
#algorithm have access to more useful variables 

volatility = (bitcoin.column("High") - bitcoin.column("Close")) /  bitcoin.column("Close") * 100
bitcoin.with_column('Volatility (%)', volatility)

percent_daily_change = bitcoin.column("Close") - bitcoin.column("Open") / bitcoin.column("Open") * 100
bitcoin.with_column('Daily Change(%)', percent_daily_change)

bitcoin 


# In[309]:


#Establishing future prices 

predicting_col = 'Close' 

daysback = 0.01 #how far in the future we'd like to predict (in our case 1% based on data sample)
future_days = int(math.ceil(daysback * bitcoin.num_rows)) 
print("We want to predict", str(future_days), "days in the future") 

future_prices = np.roll(bitcoin.column(predicting_col), bitcoin.num_rows - future_days) 

bitcoin = bitcoin.with_column("Future", future_prices) #The prices 1% in the future from any given data point

bitcoin = bitcoin.take(range(0, bitcoin.num_rows - future_days)) # no longer want the last 14 rows because they'll be false

bitcoin


# In[337]:


y = np.array(bitcoin.column("Future")) #ultimate result 

bitcoin_predictors = bitcoin.drop(["Date","Future"]) #dropping features we don't want for predicting y 

row_count = bitcoin_predictors.num_rows
col_count = bitcoin_predictors.num_columns

X = np.zeros(shape = (row_count, col_count)) # create and populate X, the features that are used to determine y 
for row in range(row_count): 
    arr = np.array(bitcoin_predictors.row(row))
    X[row] = arr

#Below, what we're doing is taking 80% of the data we have, and using it as training, develop a classifier
#20% are then used for testing once we've developed that classifier
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) 

classifier = LinearRegression() #classifier before training
classifier.fit(X_train,y_train) #trains classifier 

forecast_set = classifier.predict(X_test)

plots.scatter(forecast_set, y_test)


# In[338]:


accuracy = classifier.score(X_test, y_test) # correlation of the above scatter plot
print(accuracy)


# Using Machine Learning helped us to get an accuracy score of 84.7%. The reason that this was so much more accurate than a straight up linear regression of closing price against time, is that machine learning enbales a computer to create its own algorithms based on a multitude of variables, not just one.

# In[339]:


_ = ok.submit()

