# %%
import pandas as pd
import gzip
import json
from tqdm import tqdm
tqdm.pandas()

from nltk.corpus import stopwords
stops = set(stopwords.words('english'))



#pip install Keybert
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
def generate_sentiment(reviews:pd.core.frame.DataFrame, category:str='gift_cards')->pd.core.frame.DataFrame:
  # sentiment analysis on pretrained weights on gift cards
  sentiment_pipeline = pipeline(model="nlptown/bert-base-multilingual-uncased-sentiment")
  reviews['sentiment'] = reviews['reviewText'].progress_apply(lambda x: sentiment_pipeline(str(x)[:1000]))
  reviews['sentiment'] = reviews['sentiment'].apply(lambda x: int(x[0]['label'].split(' ')[0]))
  
  reviews.to_pickle(f'data/reviews_{category}.pkl')

  return reviews

# generate review sentiment
reviews = generate_sentiment(reviews, category)
 