import chromadb
import pandas as pd
import networkx as nx

def build_search_pipeline():
    # Define exact local paths
    data_path = r"D:\NeuroGraph_Project\Data\cleaned_disease_symptoms.csv"
    db_path = r"D:\NeuroGraph_Project\Chroma_DB"

    print("Loading symptom data for ChromaDB...")
    df = pd.read_csv(data_path)
    unique_symptoms = df['Symptom'].unique().tolist()

    # 1. Initialize persistent local vector database
    client = chromadb.PersistentClient(path=db_path)
    
    # 2. Create collection (Automatically downloads a lightweight embedding model)
    collection = client.get_or_create_collection(name="symptom_collection")
    
    # Check if already populated to skip redundant processing
    if collection.count() == 0:
        print(f"Embedding {len(unique_symptoms)} symptoms...")
        collection.add(
            documents=unique_symptoms,
            ids=[f"symp_{i}" for i in range(len(unique_symptoms))]
        )
    
    # 3. Rebuild lightweight NetworkX graph for multi-hop traversal
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Disease'], row['Symptom'], relationship="HAS_SYMPTOM")

    return collection, G

def hybrid_query(user_input, collection, G):
    print(f"\n[Patient Query]: {user_input}")
    
    # Step A: Semantic Vector Search
    results = collection.query(
        query_texts=[user_input],
        n_results=3 
    )
    
    matched_symptoms = results['documents'][0]
    print(f"-> AI Mapped Symptoms: {matched_symptoms}")

    # Step B: Graph Traversal (Multi-Hop)
    candidate_diseases = {}
    for symp in matched_symptoms:
        if G.has_node(symp):
            # Reverse traverse: Find diseases that point to this symptom
            for disease in G.predecessors(symp):
                candidate_diseases[disease] = candidate_diseases.get(disease, 0) + 1

    # Sort diseases by highest symptom match count
    sorted_diseases = sorted(candidate_diseases.items(), key=lambda x: x[1], reverse=True)
    
    print("\n-> Graph Engine Diagnostics (Top Candidates):")
    for disease, match_count in sorted_diseases[:3]:
        print(f"   {disease.title()} (Matched {match_count} symptoms via multi-hop)")

if __name__ == "__main__":
    chroma_col, kg_graph = build_search_pipeline()
    
    # Test the system with conversational natural language
    test_input = "My stomach is hurting really bad and I feel like throwing up"
    hybrid_query(test_input, chroma_col, kg_graph)