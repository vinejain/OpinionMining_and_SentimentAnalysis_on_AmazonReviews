'''
Amazon Product Reviews: Sentiment Analysis and Opinion Mining (using text mining and topic modelling)
Author: Vineet Jain
'''

# Set working directory
import os
os.getcwd()
os.chdir(r"D:\WorkRepo\AmzData")

# Load the libraries
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
import requests
from nltk.corpus import stopwords
from nltk import FreqDist
import seaborn as sns
import os, json
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
import keras
from keras.layers import Dense,LSTM
from keras.models import Sequential

'''
# Get individual category
amz = []
for line in open('Musical_Instruments_5.json', 'r'):
    amz.append(json.loads(line))    

df = pd.DataFrame(data=amz)   
'''
'''
# Get individual ASIN
df = pd.read_csv("asin_B00J4TBMVO.csv")

'''
# Merge the categories
import os, json
import pandas as pd

path_to_json = 'D:\WorkRepo\AmzData'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
df = pd.DataFrame(data=json_files)   

# Pre-Processing
df['cleaned_text'] = df['reviewText'].replace(r'\'|\"|\,|\.|\?|\+|\-|\/|\=|\(|\)|\n|"', '', regex=True)
df['cleaned_text'] = df['cleaned_text'].replace(",", " ")
df['cleaned_text'] = df['cleaned_text'].replace("  ", " ")
df['cleaned_text'] = df['cleaned_text'].str.lower()
df['cleaned_text'] = df['cleaned_text'].replace(r'<ed>','', regex = True)
df['cleaned_text'] = df['cleaned_text'].replace(r'\B<U+.*>|<U+.*>\B|<U+.*>','', regex = True)
    
# convert texts to lowercase
df['cleaned_text'] = df['cleaned_text'].str.lower()
#remove user mentions
df['cleaned_text'] = df['cleaned_text'].replace(r'^(@\w+)',"", regex=True)    
#remove_symbols
df['cleaned_text'] = df['cleaned_text'].replace(r'[^a-zA-Z0-9]', " ", regex=True)
#remove punctuations 
df['cleaned_text'] = df['cleaned_text'].replace(r'[[]!"#$%\'()\*+,-./:;<=>?^_`{|}]+',"", regex = True)
#remove_URL(x):
df['cleaned_text'] = df['cleaned_text'].replace(r'https.*$', "", regex = True)
#remove words of length 1 or 2 
df['cleaned_text'] = df['cleaned_text'].replace(r'\b[a-zA-Z]{1,2}\b','', regex=True)
#remove extra spaces in the text
df['cleaned_text'] = df['cleaned_text'].replace(r'^\s+|\s+$'," ", regex=True)

# Missing values     
df.isna().sum()
df.reviewText.fillna("",inplace = True)
df['text'] = df['reviewText'] + ' ' + df['summary']

def sent_scoring(score):
    if(int(score) == 1 or int(score) == 2 or int(score) == 3):
        return 0
    else: 
        return 1
df.overall = df.overall.apply(sent_scoring) 

stopwordsTxt = set(stopwords.words('english'))
p = list(string.punctuation)
stopwordsTxt.update(p)

def posTagger(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
    
wordLemmatize = WordNetLemmatizer()
def applyLemmatizer(text):
    wordVec = []
    for i in text.split():
        if i.strip().lower() not in stopwordsTxt:
            temp = pos_tag([i.strip()])
            word = wordLemmatize.lemmatize(i.strip(),posTagger(temp[0][1]))
            wordVec.append(word.lower())
    return " ".join(wordVec)


df.text = df.text.apply(applyLemmatizer)

# Dividing df into train and test datasets
x_train,x_test,y_train,y_test = train_test_split(df.text,df.overall,test_size = 0.2 , random_state = 0)

positive_revs = x_train[y_train[y_train == 1].index]
neagtive_revs = x_train[y_train[y_train == 0].index]
positive_revs.shape
neagtive_revs.shape


countfIdfVecec=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
countfIdfVecec_train_reviews=countfIdfVecec.fit_transform(x_train)
countfIdfVecec_test_reviews=countfIdfVecec.transform(x_test)
tfIdfVec=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
tfIdfVec_train_reviews=tfIdfVec.fit_transform(x_train)
tfIdfVec_test_reviews=tfIdfVec.transform(x_test)

logisticReg=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=0)
logisticReg_bow=logisticReg.fit(countfIdfVecec_train_reviews,y_train)
logisticReg_tfidf=logisticReg.fit(tfIdfVec_train_reviews,y_train)

#Logistic Reg
logisticReg_bow_predict=logisticReg.predict(countfIdfVecec_test_reviews)
logisticReg_tfidf_predict=logisticReg.predict(tfIdfVec_test_reviews)
logisticReg_bow_score=accuracy_score(y_test,logisticReg_bow_predict)
print("logisticReg_bow_score :",logisticReg_bow_score)
logisticReg_tfidf_score=accuracy_score(y_test,logisticReg_tfidf_predict)
print("logisticReg_tfidf_score :",logisticReg_tfidf_score)
logisticReg_bow_report=classification_report(y_test,logisticReg_bow_predict,target_names=['0','1'])
print(logisticReg_bow_report)
logisticReg_tfidf_report=classification_report(y_test,logisticReg_tfidf_predict,target_names=['0','1'])
print(logisticReg_tfidf_report)

#Naive Bayes
naiveB=MultinomialNB()
naiveB_bow=naiveB.fit(countfIdfVecec_train_reviews,y_train)
print(naiveB_bow)
naiveB_tfidf=naiveB.fit(tfIdfVec_train_reviews,y_train)
print(naiveB_tfidf)
naiveB_bow_predict=naiveB.predict(countfIdfVecec_test_reviews)
naiveB_tfidf_predict=naiveB.predict(tfIdfVec_test_reviews)
naiveB_bow_score=accuracy_score(y_test,naiveB_bow_predict)
print("naiveB_bow_score :",naiveB_bow_score)
naiveB_tfidf_score=accuracy_score(y_test,naiveB_tfidf_predict)
print("naiveB_tfidf_score :",naiveB_tfidf_score)
naiveB_bow_report = classification_report(y_test,naiveB_bow_predict,target_names = ['0','1'])
print(naiveB_bow_report)
naiveB_tfidf_report = classification_report(y_test,naiveB_tfidf_predict,target_names = ['0','1'])
print(naiveB_tfidf_report)


# Get individual ASIN
df = pd.read_csv("asin_B00J4TBMVO.csv")


# Tokenize sentences into word tokens -> each record in a separate column
from nltk.tokenize import word_tokenize
#from nltk import WhitespaceTokenizer
df['tokenized_text'] = df['cleaned_text'].apply(word_tokenize)

# Collect all tokens in one single list - 'tokens'
tokens = [item for sublist in df['tokenized_text'] for item in sublist]

# Removing stopwords from each comment
stop_words = set(stopwords.words('english'))
df['fully_cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

#Word Embeddings and Lemmatizer
from textblob import TextBlob
df['sentiment'] = df['fully_cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)  #-1 to 1
df['tokenized_text'] = df['fully_cleaned_text'].apply(word_tokenize)
df['tokenized_text'] = df['tokenized_text'].apply(lambda x: [y for y in x if not any(c.isdigit() for c in y)])

# Set values for parameters
num_features = 100    
min_word_count = 1                        
num_workers = 4       
context = 10       
                                                                                     
# word2vec
from gensim.models import word2vec
print("Training model...")
model = word2vec.Word2Vec(df['tokenized_text'], workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context)

model.init_sims(replace=True)


import numpy as np
vocab = list(model.wv.vocab)
def sentence_vector(sentence, model):
    nwords = 0
    featureV = np.zeros(100, dtype="float32")
    for word in sentence:
        if word not in vocab:
            continue
        featureV = np.add(featureV, model[word])
        nwords = nwords + 1
    if nwords > 0: 
        featureV = np.divide(featureV, nwords)
    return featureV

text_vector = df['tokenized_text'].apply(lambda x: sentence_vector(x, model))  
text_vector = text_vector.apply(pd.Series)

#text vector should vary from 0 to 1 (normalize the vector)
for x in range(len(text_vector)):
    x_min = text_vector.iloc[x].min()
    x_max = text_vector.iloc[x].max()
    X  = text_vector.iloc[x]
    i = 0
    if (x_max - x_min) == 0:
        for y in X:
            text_vector.iloc[x][i] = (1/len(text_vector.iloc[x]))
            i = i + 1
    else:
        for y in X:
            text_vector.iloc[x][i] = ((y - x_min)/(x_max - x_min))
            i = i + 1

#Scale the 'sentiment' vector
#Sentiment varies from -1 to +1

def sentiment(x):
    if x < 0.04:
        return 0
    elif x > 0.04:
        return 1
    else:
        return 0.5

text_vector[100] = df['sentiment'].apply(lambda x: sentiment(x))
#Updating the 'sentiment' column in df also
df['sentiment'] = text_vector[100]
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

#range_n_clusters = [4, 5, 6, 7, 8, 9, 10, 11]
range_n_clusters = [1,2,3, 4, 5, 6, 7, 8]
X = text_vector
n_best_clusters = 0
silhouette_best = 0
for n_clusters in range_n_clusters:
    
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
                                      #, sample_size = 5000)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    
    if silhouette_avg > silhouette_best:
        silhouette_best = silhouette_avg
        n_best_clusters = n_clusters

# get n best cluster
n_best_clusters
clusterer = KMeans(n_clusters= n_best_clusters , random_state=10)
cluster_labels = clusterer.fit_predict(X)
np.unique(cluster_labels)  

#Array of texts, the corresponding cluster number, sentiment
finaldf = pd.DataFrame({'cl_num': cluster_labels,'fully_cleaned_text': df['fully_cleaned_text'], 'cleaned_text': df['cleaned_text'], 'reviewText': df['reviewText'],'sentiment': df['sentiment']})
finaldf = finaldf.sort_values(by=['cl_num'])

df['cl_num'] = cluster_labels

dfOrdered = pd.DataFrame(df)

#Compute how many times a text has been 'retexted' - that is, how many rows in dfOrdered are identical
dfOrdered['tokenized_text'] = dfOrdered['tokenized_text'].apply(tuple)
dfUnique = dfOrdered.groupby(['reviewText', 'cleaned_text', 'fully_cleaned_text', 'sentiment','tokenized_text', 'cl_num']).size().reset_index(name="freq")
dfUnique = dfUnique.sort_values(by=['cl_num'])

dfUnique['tokenized_text'] = dfUnique['tokenized_text'].apply(list)
dfOrdered['tokenized_text'] = dfOrdered['tokenized_text'].apply(list)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X, cluster_labels)

poor_cluster_indices = []
avg_cluster_sil_score = []

for i in range(n_best_clusters):
# Aggregate the silhouette scores for samples belonging to
# cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        avgscore = (np.mean(ith_cluster_silhouette_values))   #average silhouette score for each cluster
        avg_cluster_sil_score = np.append(avg_cluster_sil_score, avgscore)
        print('Cluster',i, ':', avgscore)
        if avgscore < 0.02:
            poor_cluster_indices = np.append(poor_cluster_indices, i)
            
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
		
#remove those rows where cluster value match poor_cluster_indices 
avg_cluster_sil_score_final = []
cluster_name = np.unique(dfOrdered['cl_num'])

if (len(poor_cluster_indices)!=0):
    n_final_clusters = n_best_clusters - len(poor_cluster_indices)
    for i in poor_cluster_indices:
        dfUnique = dfUnique[dfUnique['cl_num'] != i]
    for j in cluster_name:
        if j not in poor_cluster_indices:    
            avg_cluster_sil_score_final = np.append(avg_cluster_sil_score_final, avg_cluster_sil_score[j])
            
    cluster_name = np.unique(dfUnique['cl_num'])
    
	
dfUnique['cl_num'] = abs(dfUnique['cl_num'])
dfUnique = dfUnique.sort_values(by=['cl_num'])
texts_to_consider = 'fully_cleaned_text'
final_clusters = np.unique(dfUnique['cl_num'])
print(final_clusters)


#Store all texts corresponding to each cluster in a file
for i in final_clusters:
    with open('./texts_Cluster_'+str(i)+'.txt','w') as out:
        y = ''
        for x in dfUnique[texts_to_consider][dfUnique.cl_num == i]:    
            y = y + x + '. '
        out.write(y)
        out.close()

#A combination of (Noun, adjective, cardinal number, foreign word and Verb) are being extracted now
#Extract chunks matching pattern.
        
import re
import nltk

phrases = pd.DataFrame({'extracted_phrases': [], 'cluster_num': []})
A = '(CD|JJ)/\w+\s'  #cd or jj
B = '(NN|NNS|NNP|NNPS)/\w+\s'  #nouns
C = '(VB|VBD|VBG|VBN|VBP|VBZ)/\w+\s' #verbs
D = 'FW/\w+\s'  #foreign word
patterns = ['('+A+B+')+', '('+D+B+')+','('+D+')+', '('+B+')+', '('+D+A+B+')+', 
           '('+B+C+')+', '('+D+B+C+')+', '('+B+A+B+')+', '('+B+B+C+')+'] 


def extract_phrases(tag1, tag2, sentences):
    extract_phrase = []
    for sentence in sentences:
        phrase = []
        next_word = 0
        for word, pos in nltk.pos_tag(nltk.word_tokenize(sentence)):
            if next_word == 1:
                next_word = 0
                if pos == tag2:
                    extract_phrase = np.append(extract_phrase,phrase + ' ' + word) 
            
            if pos == tag1:
                next_word = 1
                phrase = word
    return extract_phrase

for i in cluster_name:
    File = open('./texts_Cluster_'+str(i)+'.txt', 'r') #open file
    lines = File.read() #read all lines
    sentences = nltk.sent_tokenize(lines) #tokenize sentences

    for sentence in sentences: 
        f = nltk.pos_tag(nltk.word_tokenize(sentence))
        tag_seq = []
        for word, pos in f:
            tag_seq.append(pos+'/'+ word)
        X = " ".join(tag_seq)

        phrase = []
        for j in range(len(patterns)):
            if re.search(patterns[j], X):
                phrase.append(' '.join([word.split('/')[1] for word in re.search(patterns[j], X).group(0).split()]))
    
        k = pd.DataFrame({'extracted_phrases': np.unique(phrase), 'cluster_num': int(i)})
    
        phrases = pd.concat([phrases,k], ignore_index = True)

print(phrases)


#For each phrase identified replace all the substrings by the largest phrase 
#Instead of 3 different phrases, there will be only one large phrase

phrases_final = pd.DataFrame({'extracted_phrases': [], 'cluster_num': []})
for i in cluster_name:
    phrases_for_each_cluster = []
    cluster_phrases = phrases['extracted_phrases'][phrases.cluster_num == i]
    cluster_phrases = np.unique(np.array(cluster_phrases))
    for j in range(len(cluster_phrases)):
        
        phrase = cluster_phrases[j]
        updated_cluster_phrases = np.delete((cluster_phrases), j)
        if any(phrase in phr for phr in updated_cluster_phrases): 
            'y'
        else: 
            #considering phrases of length greater than 1 only
            if (len(phrase.split(' '))) > 1:
                phrases_for_each_cluster.append(phrase)
    k = pd.DataFrame({'extracted_phrases': phrases_for_each_cluster, 'cluster_num': int(i) })
    
    phrases_final = pd.concat([phrases_final,k], ignore_index = True)
	
	
#Term-frequency : For each cluster, calculate the number of times a given phrase occur in the texts of that cluster
phrases_final['term_freq'] = len(phrases_final)*[0]

for i in cluster_name:
    for phrase in phrases_final['extracted_phrases'][phrases_final.cluster_num == i]:
        texts = dfUnique[texts_to_consider][dfUnique.cl_num == i]
        for text in texts:
            if phrase in text:
                phrases_final['term_freq'][(phrases_final.extracted_phrases == phrase) & (phrases_final.cluster_num == i)] = phrases_final['term_freq'][(phrases_final.extracted_phrases == phrase) & (phrases_final.cluster_num == i)] + 1
				
				
#Document-frequency
phrases_final['doc_freq'] = len(phrases_final)*[0]


# for each phrase, compute the number of clusters that Sphrase occurs in
for phrase in phrases_final['extracted_phrases']:
    for i in cluster_name:
        all_texts = ''
        for text in dfUnique[texts_to_consider][dfUnique.cl_num == i]:
            all_texts = all_texts + text + '. ' 
        if phrase in all_texts:
            phrases_final['doc_freq'][(phrases_final.extracted_phrases == phrase) & (phrases_final.cluster_num == i)] = phrases_final['doc_freq'][(phrases_final.extracted_phrases == phrase) & (phrases_final.cluster_num == i)] + 1
        
import math
phrases_final['doc_freq'] = phrases_final['doc_freq'].apply(lambda x: math.log10(n_best_clusters/(x)) )
phrases_final['tf-idf'] = phrases_final['term_freq']*phrases_final['doc_freq']
phrases_final['diff_tf-idf'] = len(phrases_final)*[0]

narrative = pd.DataFrame({'cl_num': [], 'abstraction': []})
for i in cluster_name: 
    # arrange in descending order of tf-idf score
    phrases_final = phrases_final.sort_values(['cluster_num','tf-idf'], ascending=[1,0])
    
    #Break this distribution at a point where the difference between any consecutive phrases is maximum
    #difference between consecutive values of tf-idf 
    phrases_final['diff_tf-idf'][phrases_final.cluster_num == i] = abs(phrases_final['tf-idf'][phrases_final.cluster_num == i] - phrases_final['tf-idf'][phrases_final.cluster_num == i].shift(1))

    #The last value for each cluster will be 'NaN'. Replacing it with '0'. 
    phrases_final = phrases_final.fillna(0)
    
    phrases_final = phrases_final.reset_index(drop = True) #to avoid old index being added as a new column
    if len(phrases_final[phrases_final.cluster_num == i]) != 0:
        
        #index corresponding to the highest difference
 
        ind = (phrases_final['diff_tf-idf'][phrases_final.cluster_num == i]).idxmax()
        
        abstract = phrases_final['extracted_phrases'][:ind+1][phrases_final.cluster_num == i]
    
    
        #store the abstraction corresponding to each cluster
        k = pd.DataFrame({'cl_num': int(i), 'abstraction': abstract})
        narrative = pd.concat([narrative,k], ignore_index = True)
#Assigning polarity based on the sentiment for each text 2=negative, 1=positive, 3=neutral
dfUnique['polarity'] = np.NaN
dfUnique['polarity'][dfUnique.sentiment == 0.5] = "3"
dfUnique['polarity'][dfUnique.sentiment == 1] = "1"
dfUnique['polarity'][dfUnique.sentiment == 0] = "2"

from collections import Counter

#find the highest occurring sentiment corresponding to each text
def find_mode(a):
    b = Counter(a).most_common(3)
    mode = []; c_max = 0
    for a,c in b:
        if c>c_max:
            c_max = c
        if c_max == c:
            mode.append(a)  
    print(mode)
    mode.sort()
    print(mode)
    
    ## if mode is 3&2 i.e. neutral and negative, assign the overall sentiment for that phrase as negative, 
    ## if mode is 3&1 i.e. neutral and positive, assign the overall sentiment for that phrase as positive,
    ## if mode is 2&1 i.e. negative and positive, assign the overall sentiment for that phrase as neutal, 
    ## if mode is 3&2&1 i.e. negative, positive and neutral, assign the overall sentiment for that phrase as neutral
    
    if len(mode) == 1:
        return mode[0]
    
    elif (len(mode) == 2) & (mode[1]=='3'):
        return mode[0]
    else:
        return 3
    
#1=>+ve 2=>-ve 3=>Neutral
narrative['expression'] = -1
dfUnique = dfUnique.reset_index(drop = True)
for i in cluster_name:
    texts = dfUnique[texts_to_consider][dfUnique.cl_num == i]
    abstracts = narrative['abstraction'][narrative.cl_num == i] 
    for abst in abstracts:
        sent = []
        for text, polarity in zip(dfUnique[texts_to_consider][dfUnique.cl_num == i], dfUnique['polarity'][dfUnique.cl_num == i]):
            if abst in text:
                sent = np.append(sent, polarity)
        
        
        if len(sent)!=0:
            ## if mode is 3&2-2, 3&1-1, 2&1-3, 3&2&1 - 3
            senti = find_mode(sent)
            if senti == '2':
                sent_value = "Negative"
            elif senti == '1':
                sent_value = "Positive"
            else:
                sent_value = "Neutral"
            narrative['expression'][(narrative.abstraction == abst) & (narrative.cl_num == i)] = sent_value
        
#sudo pip install xlwt
#sudo pip3 install openpyxl
from pandas import ExcelWriter
narrative.to_csv('narrative.csv', index=False, encoding='utf-8')

#############################################################################################################
