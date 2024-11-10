import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from preprocess import preprocess

def summarize(text, n=5):
    # Preprocess the text
    original_sentences, cleaned_sentences = preprocess(text)
    
    # Create bag-of-words matrix
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(cleaned_sentences)
    
    # Calculate word frequency for each sentence
    word_counts = np.sum(bow_matrix.toarray(), axis=1)
    
    # Select top n sentences based on word frequency
    top_n_indices = np.argsort(word_counts)[-n:]
    top_n_sentences = [original_sentences[i] for i in top_n_indices]
    
    # Save the bag-of-words model
    with open('bag_of_words_storage.pkl', 'wb') as f:
        pickle.dump(bow_matrix, f)
    
    # Join sentences to create the summary
    summary = ' '.join(top_n_sentences)
    return summary
