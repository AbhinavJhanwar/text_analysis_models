# %%
from generate_keywords import clean_text
import gzip
import pandas as pd
import json
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
from nltk import ngrams

from sentence_transformers import SentenceTransformer, util
kw_model = SentenceTransformer('all-MiniLM-L6-v2')


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

category = 'luxury_beauty'
# load data by category
reviews = load_data(category)

# %%
def generateTopic(text:str, method:str='method1')->str:
    
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

    # Combine tokens & scores
    token_score_pairs = list(zip(tokens, scores))

    # Sort by decreasing score
    token_score_pairs = sorted(token_score_pairs, key=lambda x: x[1], reverse=True)

    # Output passages & scores
    '''for token, score in token_score_pairs:
        print(score, token)'''

    # generate topic using top 5 keywords
    topic = '-'.join([item[0] for item in token_score_pairs[:5]])

  else:
    topic=None

  return topic
  

# %%
# read reviews
reviews = pd.read_pickle('/home/abhinav_jhanwar_valuelabs_com/text_analysis_models/data/reviews_gift_cards.pkl')

# clean review text
reviews['clean_reviewText'] = reviews['reviewText'].apply(clean_text)

# %%
# generate topics
reviews['topic'] = reviews['clean_reviewText'].progress_apply(generateTopic, method='method1')

# %%
# method 2
# generate embedding for all keywords
# create k-means/dbscan/hdbscan model
## use elbow method to determine n clusters
## generate word cloud for each cluster and if required extend cluster size
## now using frequency of word/highest word score, generate cluster representative
### this representative can be a single word or combination or tokens
# now fetch top 5-10 keywords to generate topic
## generate cluster representative for each keyword and finally get topic
