from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from preprocess import preprocess

def summarize(text, n_clusters=5):
    # Preprocess the text
    original_sentences, cleaned_sentences = preprocess(text)
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(tfidf_matrix)
    
    # Identify the closest sentence to each cluster center
    summaries = []
    for cluster_num in range(n_clusters):
        # Find all sentences belonging to this cluster
        cluster_indices = np.where(kmeans.labels_ == cluster_num)[0]
        cluster_sentences = [original_sentences[i] for i in cluster_indices]
        
        # Find the sentence closest to the cluster center
        cluster_center = kmeans.cluster_centers_[cluster_num]
        closest_idx = cluster_indices[
            np.argmax(cosine_similarity(tfidf_matrix[cluster_indices], cluster_center.reshape(1, -1)))
        ]
        
        # Add the selected sentence to the summary
        summaries.append(original_sentences[closest_idx])
    
    # Join selected sentences to form the summary
    summary = ' '.join(summaries)
    return summary
