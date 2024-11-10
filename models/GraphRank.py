import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess

def summarize(text, n=5):
    # Preprocess the text
    original_sentences, cleaned_sentences = preprocess(text)
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
    
    # Create cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Build a graph from the similarity matrix
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Apply PageRank to the graph
    scores = nx.pagerank(nx_graph)
    
    # Sort sentences by PageRank scores
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(original_sentences)), reverse=True)
    top_n_sentences = [sentence for _, sentence in ranked_sentences[:n]]
    
    # Join sentences to create the summary
    summary = ' '.join(top_n_sentences)
    return summary
