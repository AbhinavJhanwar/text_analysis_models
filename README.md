## Installation
```
pip install .
```

## Generate Keywords
Choose from one of the four methods that suits best for your requirement- 
1. simple- General keyword extractor using given BERT model
2. candidate- Use yake candidate keywords to perform keyword extraction on BERT model
3. guided- Use seed words to find related keywords
4. embedding- Uses embeddings of document and words to generate score using cosine similarity
```
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

from keybert import KeyBERT
from text_analysis_models.generate_keywords import generate_keywords

kw_model = KeyBERT('all-distilroberta-v1')

doc = """
         Family. The fact is, there is no foundation, no secure ground, 
         upon which people may stand today if it isn't the family. It's 
         become quite clear to Morrie as he's been sick. If you don't 
         have the support and love and caring and concern that you get from
         family, you don't have much at all. Love is so supremely important. 
         As our great poet Auden said, 'Love each other or perish.'
         Say you are divorced, or living alone, or had no childern. This
         disease-what Morrie is going through-would be so much harder. He is
         not sure he could do it. Sure, people would come visit, friends, 
         associates, but it's not the same as having someone who will not 
         leave. It's not the same as having someone whom you know has an eye
         on you, is watching you the whole time.
         This is part of what a family is about, not just love, but letting 
         others know there's someone who is watching out for them. It's what 
         Morrie missed so much when his mother died-what he calls your 'spiritual
         security'-knowing that your family will be there watching out for you. 
         Nothing else will give you that. Not money. Not fame. Not work.
      """

keywords = generate_keywords(doc.lower(), kw_model, keyphrase_ngram_range=(1, 3), stop_words=stops, method='simple', highlight=True) 
print("simple method", keywords, sep='\n')

```
For example of other algorithms, check main.py file

## Generate Keyword Plots
```
from text_analysis_models.generate_keywords import generate_keywords
from text_analysis_models.plot_keywords import plot_data
from sentence_transformers import SentenceTransformer
import pandas as pd

doc = """
         Family. The fact is, there is no foundation, no secure ground, 
         upon which people may stand today if it isn't the family. It's 
         become quite clear to Morrie as he's been sick. If you don't 
         have the support and love and caring and concern that you get from
         family, you don't have much at all. Love is so supremely important. 
         As our great poet Auden said, 'Love each other or perish.'
         Say you are divorced, or living alone, or had no childern. This
         disease-what Morrie is going through-would be so much harder. He is
         not sure he could do it. Sure, people would come visit, friends, 
         associates, but it's not the same as having someone who will not 
         leave. It's not the same as having someone whom you know has an eye
         on you, is watching you the whole time.
         This is part of what a family is about, not just love, but letting 
         others know there's someone who is watching out for them. It's what 
         Morrie missed so much when his mother died-what he calls your 'spiritual
         security'-knowing that your family will be there watching out for you. 
         Nothing else will give you that. Not money. Not fame. Not work.
      """

# generate token and score
kw_model = SentenceTransformer('all-distilroberta-v1')#('all-MiniLM-L6-v2')
token_score_pairs = generate_keywords(doc.lower(), kw_model, method='embedding')
df = pd.DataFrame(token_score_pairs, columns=['keyword', 'weights'])

# plot data
plot_data(data=df.copy(), 
    keyword_column='keyword', 
    weights_column='weights',
    plot_type='phrase', 
    x_axis='count', 
    y_axis='importance', 
    bubble_size='count', 
    title_text='Keyword Importance', 
    save_file='bubble_plot', 
    sort_data='importance', min_size=0.005, 
    number_of_keywords_to_plot=20)

```

## Sentiment Generation
```
from text_analysis_models.sentiment_analysis import generate_sentiment
doc = """
      The Table looks better than the pics. 
      It is very Sturdy. The seller contacted me to ask 
      my colour preferences for the stool tapestry and
      what polish I want for my table. He did a fabulous 
      job and my table looks just the way I wanted it to! 
      Total value for money. 5 stars to the product, 
      seller and Amazon
      """

sentiment = generate_sentiment(doc)
```

## Topic Modeling
Choose from below methods-
1. method1- uses simple BERT and cosine similarity score to generate topic from top performing keywords. <b>Flaw</b>- has too many similar keywords in the final topic. Example- cute and cute dogs both can be in the topic while only cute dogs is suffice.
2. method2- uses hdbscan clustering on top of method1. <b>Benefit</b>- topics will be more wide spread and distinct. Hence, will give more idea of topics in the document.
3. method3- this is good when there are multiple documents and we are looking for uniqueness among them instead of just one document. It generates the important tokens that appear in a particular document frequently but rarely across other documents.
for more details on implementation check main.py file.
```
from text_analysis_models.generate_topic import generateTopic
doc = """
      The Table looks better than the pics. 
      It is very Sturdy. The seller contacted me to ask 
      my colour preferences for the stool tapestry and
      what polish I want for my table. He did a fabulous 
      job and my table looks just the way I wanted it to! 
      Total value for money. 5 stars to the product, 
      seller and Amazon
      """ 
topic = generateTopic(doc, method='method1')
```
TODO: other clustering methods like DBScan can be tried out and utilized as per the usecase

## References-
* Quickstart- https://maartengr.github.io/KeyBERT/guides/quickstart.html
* Pretrained Models- https://www.sbert.net/docs/pretrained_models.html
* https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads&search=sentiment