from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from preprocess import preprocess

def summarize(text, n=5):
    # Preprocess the text
    original_sentences, cleaned_sentences = preprocess(text)
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
    
    # Calculate TF-IDF score for each sentence
    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
    
    # Select top n sentences based on TF-IDF scores
    top_n_indices = np.argsort(sentence_scores)[-n:]
    top_n_sentences = [original_sentences[i] for i in top_n_indices]
    
    # Join sentences to create the summary
    summary = ' '.join(top_n_sentences)
    return summary
