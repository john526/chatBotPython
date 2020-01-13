"""
  - Description : This is a 'self learning' chatbot program
  - Install :
     - the package NLTK :
         -pip install nltk
     - the package NEWSPAPER3K
         - pip install newspaper3k
     - the package SKLEARN :
        - pip install sklearn


  - other :
    - clinic url :
       - https://www.mayoclinic.org/diseases-conditions/chronic-kidney-disease/symptoms-causes/syc-20354521

  - Unitled42.ipynb


  - At the End :
     - some question :
       - what is chronic disease ?
       - what are potential complications ?
       - who is Beyonce ?
       - hi what is chronic disease ?
       - Hi, what is chronic disease ?
"""
#import libraries 1

from newspaper import Article
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
import warnings

#Ignore any warning messages 2
warnings.filterwarnings("ignore")

#Download the packages from NLTK
nltk.download('punkt',quiet=True)
nltk.download('wordnet',quiet=True)

#Get the article URL 3
article = Article('https://www.mayoclinic.org/diseases-conditions/chronic-kidney-disease/symptoms-causes/syc-20354521')
article.download()
article.parse()
article.nlp()

corpus = article.text
#print corpus/text
#print(corpus)

#Tokenization 4
text = corpus
sent_tokens = nltk.sent_tokenize(text) #convent the text into a list of sentences

#Print the list of sentences
#print(sent_tokens)


#Create a dictionary (key:value) pair to remove punctuations 5
remove_punct_dict = dict( (punct,None) for punct in string.punctuation) #remove_punct_dict = dict( (ord(punct),None) for punct in string.punctuation)

#Print punctuation
#print(string.punctuation)
#Print dictionary
#print(remove_punct_dict)

#Create a function to return a list of lemmatized lower cas words after removing punctuation 6
def LemNormalize(text):
    return nltk.word_tokenize(text.lower().translate(remove_punct_dict))

#Print the tokenization text
#print(LemNormalize(text))

# Keyword Matching 7
#Greeting Input
GREETING_INPUTS = ["hi","hello","hola","greetings","wassup","hey"]
#Greeting response back to the user
GREETING_RESPONSES = ["howdy","hi","hey","what's good","hello","hey there"]

#Function to return random greeting response to a users greeting
def greeting(sentence):
    #if the user's input  is a greeting, then return a randomly chosen greeting response
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

#Generate the response 9
def response(user_response):

    # The users response / query 8
    #user_response = 'What is chronic kidney desease'

    user_response = user_response.lower()  # Make the response lower case
    # Print the users query / response
    #print(user_response)

    # Set the chatbot response to an empty string
    robo_response = ''

    # Append the users response to the sentence list
    sent_tokens.append(user_response)
    # Print the sentence list after appending the users response
    #print(sent_tokens)

    # Create a TfidVectorizer Object
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')

    # Convert the text to a matrix of TF-IDF features
    tfidf = TfidfVec.fit_transform(sent_tokens)
    # Print the TFIDF features
    #print(tfidf)

    # Get the measure of similarity (similarity scores)
    vals = cosine_similarity(tfidf[-1], tfidf)
    # Print the similary scores
    #print(vals)

    # Get the index of the most similar text/sentence to the users response
    idx = vals.argsort()[0][-2]

    # Reduce the dimensionality of vals
    flat = vals.flatten()

    ##Sort the list in ascending order
    flat.sort()

    # Get the most similar score to the users response
    score = flat[-2]
    # Print similarity score
    #print(score)

    # If the variable 'score' is a 0 then their is no text similar to the users response

    if (score == 0):
        robo_response = robo_response + " I'm apologize, I don't understand."
    else:
        robo_response = robo_response + sent_tokens[idx]

    # Print the chat bot response
    #print(robo_response)

    #Remove the users response from the tokens list
    sent_tokens.remove(user_response)

    #Return
    return robo_response

# 10
flag = True
print("DOCBot : I'm Doctor Bot or DOCBot for short. I will answer your queries about Chronic Kidney Disease. If you want Exit, type Bye!")
while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response == 'thanks' or user_response == 'thank you'):
            print("DOCBot : You are welcome !")
        else:
            if(greeting(user_response) != None):
                print("DOCBot : "+greeting(user_response))
            else:
                print("DOCBot : "+response(user_response))
                #sent_tokens.remove(user_response)
    else:
        flag = False
        print("DOCBot : Chat with your later ! ")