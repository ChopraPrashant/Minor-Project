from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from preprocess import preprocess

def summarize(text, n=5):
    # Preprocess the text
    original_sentences, cleaned_sentences = preprocess(text)
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
    
    # Apply LSA with SVD
    svd = TruncatedSVD(n_components=1)
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    
    # Calculate scores for each sentence based on LSA component
    sentence_scores = lsa_matrix[:, 0]
    
    # Select top n sentences based on LSA scores
    top_n_indices = np.argsort(sentence_scores)[-n:]
    top_n_sentences = [original_sentences[i] for i in top_n_indices]
    
    # Join sentences to create the summary
    summary = ' '.join(top_n_sentences)
    return summary
