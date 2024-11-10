import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Clean and tokenize each sentence
    cleaned_sentences = []
    for sentence in sentences:
        # Remove non-alphabetic characters
        sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
        # Tokenize and remove stopwords
        words = word_tokenize(sentence.lower())
        words = [word for word in words if word not in stop_words]
        cleaned_sentences.append(' '.join(words))
    
    return sentences, cleaned_sentences  # Return both original and cleaned
