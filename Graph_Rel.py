import pandas as pd
import networkx as nx

def build_knowledge_graph():
    data_path = r"D:\NeuroGraph_Project\Data\cleaned_disease_symptoms.csv"
    
    print("Loading cleaned dataset...")
    df = pd.read_csv(data_path)

    # Initialize Directed Graph
    G = nx.DiGraph()

    print("Constructing the network nodes and edges...")
    for _, row in df.iterrows():
        disease = row['Disease']
        symptom = row['Symptom']

        # Add nodes
        G.add_node(disease, label="Disease")
        G.add_node(symptom, label="Symptom")

        # Connect Disease -> HAS_SYMPTOM -> Symptom
        G.add_edge(disease, symptom, relationship="HAS_SYMPTOM")

    return G

def export_graph_to_csv(G, output_path):
    print("\nExporting graph relationships to CSV...")
    
    # Extract all edges from the NetworkX graph into a Pandas DataFrame
    edge_df = nx.to_pandas_edgelist(G)
    
    # Rename the default columns (source, target) to our specific entities
    edge_df.rename(columns={'source': 'Disease', 'target': 'Symptom'}, inplace=True)
    
    # Save it to the Data folder
    edge_df.to_csv(output_path, index=False)
    print(f"Graph relationships saved to: {output_path}")

if __name__ == "__main__":
    kg_network = build_knowledge_graph()
    
    print("Structural Base Complete: ")
    print(f"Total Unique Nodes (Diseases + Symptoms): {kg_network.number_of_nodes()}")
    print(f"Total Connections (Edges): {kg_network.number_of_edges()}")
    
    # Save the internal graph out to a readable CSV
    output_csv = r"D:\NeuroGraph_Project\Data\graph_relationships.csv"
    export_graph_to_csv(kg_network, output_csv)   