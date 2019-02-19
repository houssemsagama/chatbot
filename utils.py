import nltk
import pickle
import re
import numpy as np
import gensim
from gensim.models import KeyedVectors

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'embeddings.pkl',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    embeds = pickle.load(open(embeddings_path, 'rb'))
    
    return embeds

   
    
   
    raise NotImplementedError(
        "Open utils.py and fill with your code. In case of Google Colab, download"
        "(https://github.com/hse-aml/natural-language-processing/blob/master/project/utils.py), "
        "edit locally and upload using '> arrow on the left edge' -> Files -> UPLOAD")


def question_to_vec(question, embeddings, dim=300):
    if question == "":
        return np.zeros(dim)
    t = np.array([embeddings[i]
                  for i in question.split() if i in embeddings.keys()])
    if len(t) == 0:
        return np.zeros(dim)

return(t.mean(axis=0))
    """Transforms a string to an embedding by averaging word embeddings."""
    
    
    raise NotImplementedError(
        "Open utils.py and fill with your code. In case of Google Colab, download"
        "(https://github.com/hse-aml/natural-language-processing/blob/master/project/utils.py), "
        "edit locally and upload using '> arrow on the left edge' -> Files -> UPLOAD")


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
