import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

def prepare_graph_data(csv_path):
    print("Loading graph relationships...")
    df = pd.read_csv(csv_path)

    # Map string names to unique integer IDs
    unique_nodes = pd.concat([df['Disease'], df['Symptom']]).unique()
    node_to_id = {node: i for i, node in enumerate(unique_nodes)}
    id_to_node = {i: node for i, node in enumerate(unique_nodes)}

    source_nodes = [node_to_id[disease] for disease in df['Disease']]
    target_nodes = [node_to_id[symptom] for symptom in df['Symptom']]
    
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    # Simple Identity matrix for initial node features
    num_nodes = len(unique_nodes)
    x = torch.eye(num_nodes, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    return data, node_to_id, id_to_node

class SimpleGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def train_and_extract_embeddings():
    data_path = r"D:\NeuroGraph_Project\Data\graph_relationships.csv"
    data, node_to_id, id_to_node = prepare_graph_data(data_path)

    print("Initializing GCN Model and Optimizer...")
    model = SimpleGCN(num_features=data.num_nodes, hidden_dim=64, output_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    print("\n--- Training GNN (100 Epochs) ---")
    model.train()
    for epoch in range(1, 101):
        optimizer.zero_grad()
        
        # 1. Forward pass
        z = model(data.x, data.edge_index)
        
        # 2. Calculate Positive Loss (Real edges)
        pos_edge_index = data.edge_index
        pos_out = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
        pos_loss = criterion(pos_out, torch.ones(pos_edge_index.size(1)))
        
        # 3. Calculate Negative Loss (Fake/Random edges)
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index, 
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )
        neg_out = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
        neg_loss = criterion(neg_out, torch.zeros(neg_edge_index.size(1)))
        
        # 4. Backpropagation
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Total Loss: {loss.item():.4f}")

    print("\nTraining complete! Extracting optimized embeddings...")
    model.eval()
    with torch.no_grad():
        final_embeddings = model(data.x, data.edge_index)

    # Map the numerical tensors back to the string entities
    node_embeddings = {id_to_node[i]: final_embeddings[i].numpy() for i in range(data.num_nodes)}
    
    # Save to disk
    output_path = r"D:\NeuroGraph_Project\Data\node_embeddings.npy"
    np.save(output_path, node_embeddings)
    print(f"Success! {len(node_embeddings)} mathematically optimized embeddings saved to: {output_path}")

if __name__ == "__main__":
    train_and_extract_embeddings()