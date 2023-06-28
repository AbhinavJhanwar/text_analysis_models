# %%
import yake
import re, string

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
from nltk import ngrams

from sentence_transformers import util

# %%
def clean_text(text:str)->str:
  # lower encoding the text
  text = text.lower()
  # remove punctuations
  text = re.sub('['+string.punctuation+']+', ' ', text)     
  # Remove double whitespace                       
  text = re.sub('\s+\s+', ' ', text)      
  # Remove \ slash
  text = re.sub(r'\\', ' ', text)         
  # Remove / slash
  text = re.sub(r'\/', ' ', text)   
  
  return text      

def generate_keywords(doc, kw_model, keyphrase_ngram_range=(1, 3), stop_words='english', method='simple', seed_keywords=[], highlight=False):
      """Function to generate keywords.
      
      Method- simple:
            General keyword extractor using given BERT model
      Method- candidate:
            In some cases, one might want to be using candidate keywords 
            generated by other keyword algorithms or retrieved from a 
            select list of possible keywords/keyphrases. 
            In KeyBERT, you can easily use those candidate keywords to 
            perform keyword extraction.
      Method- guided:
            Guided KeyBERT is similar to Guided Topic Modeling in that 
            it tries to steer the training towards a set of seeded terms. 
            When applying KeyBERT it automatically extracts the most 
            related keywords to a specific document. However, there are
            times when stakeholders and users are looking for specific 
            types of keywords. For example, when publishing an article 
            on your website through contentful, you typically already 
            know the global keywords related to the article. However, 
            there might be a specific topic in the article that you would 
            like to be extracted through the keywords. To achieve this, 
            we simply give KeyBERT a set of related seeded keywords 
            (it can also be a single one!) and search for keywords that 
            are similar to both the document and the seeded keywords.
      Method- embedding:
            This method utilizes BERT model to generate embeddings for 
            the document and the words, then generates the keywords from
            those embeddings using cosine similarity and provides a score
            

      Args:
          doc (_type_): _description_
          kw_model (_type_): _description_
          keyphrase_ngram_range (tuple, optional): _description_. Defaults to (1, 3).
          stop_words (str, optional): Choose from nltk, english. Defaults to 'english'.
          method (str, optional): _description_. Defaults to 'simple'.
          seed_keywords (list, optional): _description_. Defaults to [].
          highlight (bool, optional): _description_. Defaults to False.

      Returns:
          _type_: _description_
      """

      if stop_words=='nltk':
            stop_words = stops
      
      # method 1
      if method == 'simple':
            # directly generate keywords from bert model
            keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=keyphrase_ngram_range, stop_words=stop_words, highlight=highlight)
            return keywords

      # method 2
      elif method == 'candidate':
            # generate keywords using candidates from other algorithm
            # Create candidates
            kw_extractor = yake.KeywordExtractor(top=100)
            candidates = kw_extractor.extract_keywords(doc)
            candidates = [candidate[0] for candidate in candidates]

            # Pass candidates to KeyBERT
            keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=keyphrase_ngram_range, stop_words=stop_words, highlight=highlight, candidates=candidates)
            return keywords
      
      # method 3
      elif method == 'guided':
            keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=keyphrase_ngram_range, stop_words=stop_words, highlight=highlight, seed_keywords=seed_keywords)
            return keywords

      # method 4
      elif method == 'embedding':
            doc = clean_text(doc)

            # remove stop words
            text1 = ' '.join([token for token in doc.split() if token not in stops])

            if len(text1)>2:
                  # generate n grams
                  tokens = []
                  for i in range(1, 4):
                        tokens = tokens + list(set([' '.join(tokens) for tokens in ngrams(text1.split(), i)]))

                  # keep only n grams that are present in actual sentence
                  tokens = [token for token in tokens if token in doc]

                  # generate embedding
                  doc_emb = kw_model.encode(doc)
                  token_emb = kw_model.encode(tokens)

                  # Compute cosine similarity score between tokens and document embeddings
                  scores = util.cos_sim(doc_emb, token_emb)[0].cpu().tolist()

                  # Combine tokens & scores
                  token_score_pairs = list(zip(tokens, scores))

                  # Sort by decreasing score
                  token_score_pairs = sorted(token_score_pairs, key=lambda x: x[1], reverse=True)

                  return token_score_pairs
            else:
                  return "No keywords"
