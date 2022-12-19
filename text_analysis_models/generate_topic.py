# %%
from text_analysis_models.generate_keywords import clean_text
import gzip
import pandas as pd
import json
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
from nltk import ngrams

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import hdbscan
import umap

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

# %%
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

# %%
def generateTopic(text:str, method:str='method1')->str:
  """_summary_

  Args:
      text (str): _description_
      method (str, optional): _description_. Defaults to 'method1'.

      method1: simple topic modeling based on the keyword score from
              BERT model and cosine similarity between token and doc
      method2: on top of method 1 a clustering technique hdbscan is 
              applied to cluster similar tokens and include only one
              token from each cluster

  Returns:
      str: _description_
  """
    
  text = text.lower()

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

    # generate embedding
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
      topic = '-'.join([item[0] for item in token_score_pairs[:5]])
    
    elif method=='method2':
      df_embedding = pd.DataFrame(token_emb)
      df_embedding['tokens'] = tokens
      train_data = df_embedding.drop_duplicates()

      reducer = umap.UMAP(random_state=42, n_components=2)
      embedding = reducer.fit_transform(df_embedding[list(range(0, 768))])

      # Building the clustering model
      clustering_model = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)

      # Training the model and Storing the predicted cluster labels 
      clusterer = clustering_model.fit(embedding)
      train_data['tokens'] = tokens
      train_data['cluster']  = clusterer.labels_
      train_data['scores'] = scores

      # sort as per scores
      train_data = train_data.sort_values('scores', ascending=False)[['tokens', 'scores', 'cluster']]
      
      # separate cluster -1
      temp = train_data[train_data.cluster==-1]
      train_data = train_data[train_data.cluster!=-1]

      # keep only one value from each cluster
      train_data = train_data.drop_duplicates(['cluster'])
      
      # concat cluster -1 and sort as per scores 
      train_data = pd.concat([train_data, temp]).sort_values('scores', ascending=False)
      
      # for cluster in train_data.cluster.unique():
      #   print(cluster, train_data[train_data.cluster==cluster]['tokens'].unique())

      # generate topic using top 5 keywords
      topic = '-'.join(train_data.tokens.iloc[:5].tolist())

    return topic

  else:
    return None
# method 2
# generate embedding for all keywords
# create k-means/dbscan/hdbscan model
## use elbow method to determine n clusters
## generate word cloud for each cluster and if required extend cluster size
## now using frequency of word/highest word score, generate cluster representative
### this representative can be a single word or combination or tokens
# now fetch top 5-10 keywords to generate topic
## generate cluster representative for each keyword and finally get topic
