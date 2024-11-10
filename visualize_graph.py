# visualize_graph.py
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess  # Assumes a preprocess.py file for text cleaning

def create_summary_graph(summary, by_sentences=True):
    
    # Preprocess the summary to get sentences or terms
    sentences, _ = preprocess(summary)
    
    # Use sentences or terms as nodes in the graph
    if by_sentences:
        nodes = sentences
    else:
        # Use key terms extracted via TF-IDF
        vectorizer = TfidfVectorizer(max_features=20)
        vectorizer.fit_transform([summary])
        nodes = vectorizer.get_feature_names_out()
    
    # Generate TF-IDF matrix and compute cosine similarity
    tfidf_matrix = TfidfVectorizer().fit_transform(nodes)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Initialize an undirected graph
    G = nx.Graph()
    
    # Add nodes and edges based on a similarity threshold
    for i, node in enumerate(nodes):
        G.add_node(i, label=node)
        for j in range(i+1, len(nodes)):
            if similarity_matrix[i, j] > 0.3:  # Set similarity threshold
                G.add_edge(i, j, weight=similarity_matrix[i, j])
    
    return G, nodes

def plot_graph(G, nodes):
    
    pos = nx.spring_layout(G, k=0.5, seed=42)
    labels = {i: nodes[i] for i in range(len(nodes))}
    
    # Draw nodes with gradient color based on degree
    degrees = [G.degree(n) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=degrees, cmap=plt.cm.viridis, alpha=0.8)
    
    # Draw edges with varying thickness based on weight
    edges = G.edges(data=True)
    weights = [edge_data['weight'] for _, _, edge_data in edges]
    nx.draw_networkx_edges(G, pos, edge_color=weights, edge_cmap=plt.cm.Blues, width=[3 * w for w in weights], alpha=0.6)
    
    # Draw labels for nodes and edges
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color="black", font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): f"{w:.2f}" for i, j, w in G.edges(data="weight")},
                                 font_color="gray", font_size=8)
    
    # Title and display
    plt.figure(figsize=(10, 8))
    plt.title("Enhanced Summary Graph Visualization")
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label="Node Degree")
    plt.show()
