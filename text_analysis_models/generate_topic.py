# %%
from text_analysis_models.generate_keywords import clean_text
from typing import Union
import gzip
import pandas as pd
import json
import math
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
from nltk import ngrams

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import hdbscan
import umap

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util
kw_model = SentenceTransformer('all-distilroberta-v1')#('all-MiniLM-L6-v2')


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def load_data(category:str='gift_cards')->pd.core.frame.DataFrame:
  
  if category=='gift_cards':
    # gift cards
    reviews = getDF('Gift_Cards.json.gz')
    
  elif category=='luxury_beauty':
    # beauty products
    reviews = getDF('Luxury_Beauty.json.gz')
    
  reviews = reviews[['asin', 'overall', 'reviewText']]

  # dropna
  reviews.dropna(inplace=True)

  # extract random sample of 1000 reviews
  reviews = reviews.sample(1000).reset_index(drop=True)

  return reviews

#category = 'luxury_beauty'
# load data by category
#reviews = load_data(category)

def generate_topic_method3(texts:list[str], clusterer:str='hdbscan', clusters:int=10, number_of_topics:int=5)->list[str]:
  for i, text in enumerate(texts):
    # text cleaning
    text = text.lower()
    text = clean_text(text)
    # remove stop words
    text1 = ' '.join([token for token in text.split() if token not in stops])
    texts[i] = text1
  texts = [text for text in texts if len(text)>0]

  # generate embedding for documents
  doc_emb = kw_model.encode(texts)

  # creat dataframe for linking embedding with document
  df_embedding = pd.DataFrame(doc_emb)
  df_embedding['doc'] = texts
  train_data = df_embedding.drop_duplicates()

  if clusterer=='hdbscan':
    # create clustering of all the documents
    reducer = umap.UMAP(random_state=42, n_components=2)
    embedding = reducer.fit_transform(train_data[list(range(0, doc_emb.shape[1]))])

    # Building the clustering model
    clustering_model = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)

    # Training the model and Storing the predicted cluster labels 
    clusterer = clustering_model.fit(embedding)
    train_data['cluster']  = clusterer.labels_

  elif clusterer=='kmeans':
    # rescaling data
    scale = StandardScaler()
    scaled_data = scale.fit_transform(train_data[list(range(0, doc_emb.shape[1]))])

    # Using ELBOW Method to figure out number of clusters
    inertia=[]
    silhouetteScore = []
    n_clusters = len(train_data)
    bin_ = max(1, n_clusters//10)
    print(train_data.shape)
    for i in tqdm(range(2, n_clusters, bin_)):
      kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
      kmeans.fit(scaled_data)
      inertia.append(kmeans.inertia_)
      silhouetteScore.append(silhouette_score(scaled_data, kmeans.predict(scaled_data)))

    # plot inertia and silhoutte score
    fig, ax1 = plt.subplots(figsize=(8, 5))
    #fig.text(0.1, 1, 'Skipping ', fontfamily='serif', fontsize=12, fontweight='bold')
    fig.text(0.1, 0.95, 'We want to select a point where Inertia is low & Silhouette Score is high, and the number of clusters is not overwhelming for the business.',
          fontfamily='serif',fontsize=10)
    fig.text(1.4, 1, 'Inertia', fontweight="bold", fontfamily='serif', fontsize=15, color='#244747')
    fig.text(1.51, 1, "|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
    fig.text(1.53, 1, 'Silhouette Score', fontweight="bold", fontfamily='serif', fontsize=15, color='#91b8bd')

    ax1.plot(range(2, n_clusters, bin_), inertia, '-', color='#244747', linewidth=5)
    ax1.plot(range(2, n_clusters, bin_), inertia, 'o', color='#91b8bd')
    ax1.set_ylabel('Inertia')

    ax2 = ax1.twinx()
    ax2.plot(range(2, n_clusters, bin_), silhouetteScore, '-', color='#91b8bd', linewidth=5)
    ax2.plot(range(2, n_clusters, bin_), silhouetteScore, 'o', color='#244747', alpha=0.8)
    ax2.set_ylabel('Silhouette Score')

    plt.xlabel('Number of clusters')
    plt.show()

    model = KMeans(n_clusters=clusters, init='k-means++', random_state=0, algorithm='elkan')
    y = model.fit_predict(scaled_data)
    train_data['cluster']  = y
    
  train_data = train_data[['doc', 'cluster']]

  # join train_data based on clusters excluding -1
  temp_data = train_data[train_data.cluster==-1]
  train_data = train_data[train_data.cluster!=-1]
  train_data['new_doc'] = train_data.groupby('cluster')['doc'].transform(lambda x: ' '.join(x))
  train_data = train_data.drop_duplicates(['new_doc'])
  train_data['doc'] = train_data['new_doc']
  del train_data['new_doc']

  # add cluster -1
  train_data = pd.concat([train_data, temp_data])

  # get total number of unique clusters
  total_clusters = len(train_data)

  # get total words in each cluster
  train_data['cluster_total_word_count'] = train_data['doc'].apply(lambda x: len(x.split(' ')))

  # get all words frequency in each cluster
  # get tokens in each document
  train_data['tokens'] = train_data['doc'].apply(lambda x: x.split(' '))
  train_data = train_data.explode('tokens')

  train_data['word_frequency_in_cluster'] = train_data.groupby(['tokens', 'cluster']).transform('count')['doc']
  train_data = train_data.drop_duplicates(['tokens', 'cluster'])

  # get word frequency across all clusters
  train_data['word_frequency_across_cluster'] = train_data.groupby('tokens').transform('count')['doc']

  train_data['importance_score'] = (train_data['word_frequency_in_cluster']/train_data['cluster_total_word_count'])
  train_data['temp'] = train_data['word_frequency_across_cluster'].apply(lambda x: math.log(total_clusters/x))
  train_data['importance_score'] = train_data['importance_score']*train_data['temp']
  del train_data['temp']

  # for each document generate token score and bring out important topics
  topics = []
  for text in texts:
        token_score = {}
        tokens = text.split(' ')
        for token in tokens:
              token_score[token] = train_data[train_data['tokens']==token]['importance_score'].mean()
        token_score = sorted(token_score.items(), key=lambda x: x[1], reverse=True)
        if len(token_score)>number_of_topics:
              topic = '-'.join([item[0] for item in token_score[:number_of_topics]])
        else:
              topic = '-'.join([item[0] for item in token_score])
        topics.append(topic)
  return topics

def generateTopic(original_text:Union[str, list[str]], method:str='method1', 
                  clusterer:str='hdbscan', clusters:int=10, number_of_topics:int=5)->Union[str, list[str]]:
  """_summary_

  Args:
      text (str): input text to generate topic
      method (str, optional): _description_. Defaults to 'method1'.

      method1: simple topic modeling based on the score of keyword using
              BERT model and cosine similarity between token and doc
      method2: on top of method 1, a clustering technique hdbscan is 
              applied to cluster similar tokens and include only one
              token from each cluster
      method3: method 3 is similar to method 2, instead of clustering
              the tokens, it clusters the sentences separated by '.'.
              Instead of using cosine similarity it generates an importance 
              score for each token using below formula-
              importance = (word count in current cluster/total word count in current cluster)*
                            log(total number of clusters/word count across clusters)
              In short a token appearing across clusters reduces its importance,
              while its appearance in current cluster increases its importance.
              Now for each token in each sentence an importance score is available
              and accordingly topic is generated.

  Returns:
      str: _description_
  """
  
  # clean text and generate a list
  if type(original_text)==str:
    # text cleaning
    text = original_text.lower()
    text = clean_text(text)

    # remove stop words
    text1 = ' '.join([token for token in text.split() if token not in stops])

    if len(text1)>2:
      # generate n grams
      tokens = []
      for i in range(1, 4):
        tokens = tokens + list(set([' '.join(tokens) for tokens in ngrams(text1.split(), i)]))

      # keep only n grams that are present in actual sentence
      tokens = [token for token in tokens if token in text]

      # generate embedding for document and tokens
      doc_emb = kw_model.encode(text)
      token_emb = kw_model.encode(tokens)

      # Compute cosine similarity score between tokens and document embeddings
      scores = util.cos_sim(doc_emb, token_emb)[0].cpu().tolist()

      if method=='method1':
        # Combine tokens & scores
        token_score_pairs = list(zip(tokens, scores))

        # Sort by decreasing score
        token_score_pairs = sorted(token_score_pairs, key=lambda x: x[1], reverse=True)

        # Output passages & scores
        # for token, score in token_score_pairs:
        #     print(score, token)

        # generate topic using top 5 keywords
        topic = '-'.join([item[0] for item in token_score_pairs[:number_of_topics]])
      
      elif method=='method2':
        df_embedding = pd.DataFrame(token_emb)
        df_embedding['tokens'] = tokens
        df_embedding['scores'] = scores
        train_data = df_embedding.drop_duplicates(['tokens'])

        if clusterer == 'hdbscan':
          reducer = umap.UMAP(random_state=42, n_components=2)
          embedding = reducer.fit_transform(train_data[list(range(0, doc_emb.shape[0]))])

          # Building the clustering model
          clustering_model = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)

          # Training the model and Storing the predicted cluster labels 
          clusterer = clustering_model.fit(embedding)
          train_data['cluster']  = clusterer.labels_

        elif clusterer == 'kmeans':
          # rescaling data
          scale = StandardScaler()
          scaled_data = scale.fit_transform(train_data[list(range(0, doc_emb.shape[0]))])

          # Using ELBOW Method to figure out number of clusters
          inertia=[]
          silhouetteScore = []
          n_clusters = len(train_data)
          for i in range(2, n_clusters):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
            kmeans.fit(scaled_data)
            inertia.append(kmeans.inertia_)
            silhouetteScore.append(silhouette_score(scaled_data, kmeans.predict(scaled_data)))

          # plot inertia and silhoutte score
          fig, ax1 = plt.subplots(figsize=(8, 5))
          #fig.text(0.1, 1, 'Skipping ', fontfamily='serif', fontsize=12, fontweight='bold')
          fig.text(0.1, 0.95, 'We want to select a point where Inertia is low & Silhouette Score is high, and the number of clusters is not overwhelming for the business.',
                fontfamily='serif',fontsize=10)
          fig.text(1.4, 1, 'Inertia', fontweight="bold", fontfamily='serif', fontsize=15, color='#244747')
          fig.text(1.51, 1, "|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
          fig.text(1.53, 1, 'Silhouette Score', fontweight="bold", fontfamily='serif', fontsize=15, color='#91b8bd')

          ax1.plot(range(2, n_clusters), inertia, '-', color='#244747', linewidth=5)
          ax1.plot(range(2, n_clusters), inertia, 'o', color='#91b8bd')
          ax1.set_ylabel('Inertia')

          ax2 = ax1.twinx()
          ax2.plot(range(2, n_clusters), silhouetteScore, '-', color='#91b8bd', linewidth=5)
          ax2.plot(range(2, n_clusters), silhouetteScore, 'o', color='#244747', alpha=0.8)
          ax2.set_ylabel('Silhouette Score')

          plt.xlabel('Number of clusters')
          plt.show()

          model = KMeans(n_clusters=clusters, init='k-means++', random_state=0, algorithm='elkan')
          y = model.fit_predict(scaled_data)
          train_data['cluster']  = y

        # sort as per scores
        train_data = train_data.sort_values('scores', ascending=False)[['tokens', 'scores', 'cluster']]
        # separate cluster -1
        temp = train_data[train_data.cluster==-1]
        train_data = train_data[train_data.cluster!=-1]
        # keep only one value from each cluster, as they are sorted by score already hence 
        # top keyword from each cluster will remain in the dataframe
        train_data = train_data.drop_duplicates(['cluster'])
        # concat cluster -1 and sort as per scores 
        train_data = pd.concat([train_data, temp]).sort_values('scores', ascending=False)
        # for cluster in train_data.cluster.unique():
        #   print(cluster, train_data[train_data.cluster==cluster]['tokens'].unique())

        # generate topic using top 5 keywords
        topic = '-'.join(train_data.tokens.iloc[:number_of_topics].tolist())
    
      elif method=='method3':
         texts = original_text.split('.')
         topic = generate_topic_method3(texts, clusterer=clusterer, clusters=clusters, number_of_topics=number_of_topics)
      
      # return topic generated from any of the methods
      return topic
    
    # text if not longer than 2 characters then return as it is
    else:
       return text
    
  elif type(original_text)==list:
    if method=='method3':
      topics = generate_topic_method3(original_text, clusterer=clusterer, clusters=clusters, number_of_topics=number_of_topics)

    # return error if method is incorrect
    else:
      return 'error- incorrect method'  
    
    # return topics from list
    return topics
  
  # return error if datatype of input is incorrect
  else:
    return 'error- incorrect data type'

