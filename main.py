
from keybert import KeyBERT
from text_analysis_models.generate_keywords import generate_keywords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
      

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

keywords = generate_keywords(doc.lower(), kw_model, keyphrase_ngram_range=(1, 3), stop_words=stops, method='candidate', highlight=True) 
print("candidate method", keywords, sep='\n')

# Define seeded terms
seed_keywords = ["family", "knowledge", "children"]
keywords = generate_keywords(doc.lower(), kw_model, keyphrase_ngram_range=(1, 3), stop_words=stops, method='guided', seed_keywords=seed_keywords, highlight=True) 
print("guided method", keywords, sep='\n')

