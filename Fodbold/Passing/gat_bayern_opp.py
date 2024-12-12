import torch
import torch.nn.functional
import torch_geometric.nn 
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import pandas as pd
import networkx as nx
import numpy as np
import tqdm
import random
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error



pkl_graphs = pd.read_pickle('/Users/morten/Desktop/p5 kode/5-semester/Momentum graphs.pkl')

# Initialize unique nodes
unique_nodes_bayern = []
unique_nodes_opp = []
delete_idx = []

# Collect unique nodes and filter out invalid indices
for idx, graph in pkl_graphs.items():
    if idx.endswith('45'):  # Remove unwanted graphs
        delete_idx.append(idx)
        continue
    for key, i in graph.items():
        if key == 'Bayer Leverkusen':
            unique_nodes_bayern.extend(node for node in i.nodes() if node not in unique_nodes_bayern)
        elif key == 'Opponent':
            unique_nodes_opp.extend(node for node in i.nodes() if node not in unique_nodes_opp)

# Create mapping from nodes to indices and back
idx_to_pos_bayer = dict(enumerate(unique_nodes_bayern))
pos_to_idx_bayer = {pos: idx for idx, pos in idx_to_pos_bayer.items()}

idx_to_pos_opp = dict(enumerate(unique_nodes_opp))
pos_to_idx_opp = {pos: idx for idx, pos in idx_to_pos_opp.items()}

unique_nodes_combined = unique_nodes_bayern + unique_nodes_opp
delete_idx = []

# Create mapping from nodes to indices and back for combined nodes
idx_to_pos_combined = dict(enumerate(unique_nodes_combined))
pos_to_idx_combined = {pos: idx for idx, pos in idx_to_pos_combined.items()}

# Combine data for both teams
pyg_data_combined = []
for idx, graphs in pkl_graphs.items():
    if idx in delete_idx:
        continue

    combined_graph = nx.Graph()  # Initialize an empty graph to combine both teams
    momentum = None  # To store the momentum value (for both teams)

    for key, graph in graphs.items():
        # Calculate centrality measures
        closeness = nx.closeness_centrality(graph)
        betweenness = nx.betweenness_centrality(graph)
        pagerank = nx.pagerank(graph, weight='weight')
        centrality_list = [closeness, betweenness, pagerank]

        adj_dict = nx.to_dict_of_dicts(graph)
        
        # Create a combined adjacency vector for both teams
        for node in graph.nodes():
            adj_vect = np.zeros((len(unique_nodes_combined)))
            players = adj_dict[node]
            for neighbor, value in players.items():
                if neighbor in pos_to_idx_combined:
                    adj_vect[pos_to_idx_combined[neighbor]] = value.get('weight', 0)
            adj_vect = torch.tensor(adj_vect, dtype=torch.float)

            centrality_vect = torch.tensor(
                [measure.get(node, 0) for measure in centrality_list], dtype=torch.float
            )
            node_features = torch.cat((adj_vect, centrality_vect), dim=-1)
            
            # Combine this node's data into the combined graph
            if node not in combined_graph:
                combined_graph.add_node(node)
            combined_graph.nodes[node]['x'] = node_features

        # Add edges to the combined graph
        # for u, v, data in graph.edges(data=True):
        #     if not combined_graph.has_edge(u, v):  # Avoid duplicating edges
        #         combined_graph.add_edge(u, v, **data)

        # Check if momentum is present in Bayer Leverkusen's graph and extract it
        if key == 'Bayer Leverkusen':
            if 'momentum' in graph.graph:
                momentum = graph.graph['momentum']  # Extract the momentum value
            else:
                momentum = torch.zeros(1)  # If no momentum is available, set to default

    # If momentum is found, apply it to the entire combined graph
    if momentum is not None:
        combined_graph.graph['momentum'] = momentum
    try:
        # Convert the combined NetworkX graph to PyTorch Geometric data
        data = from_networkx(combined_graph)
        pyg_data_combined.append(data)
    except Exception as e:
        print(f"Error converting graph: {e}")
len(pyg_data_combined)


    
# Normalize momentum values
momentum_values = []
for data in pyg_data_combined:
    momentum_values.append(data.momentum)
momentum_min = min(momentum_values)
momentum_max = max(momentum_values)

for data in pyg_data_combined:
    normalized_momentum = (data.momentum - momentum_min) / (momentum_max - momentum_min)
    scaled_momentum = 2 * normalized_momentum - 1  
    data.momentum = scaled_momentum
    

    
train_idx, test_idx = train_test_split(range(len(pyg_data_combined)), test_size=0.2, random_state=42)
train_data = [pyg_data_combined[idx] for idx in train_idx]
dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

class GAT(torch.nn.Module):
    def __init__(self, alpha, input_dim, hidden_dim, output_dim, num_heads, dropout=0.4):
        super(GAT, self).__init__()
        self.dropout_rate = dropout
        
        self.layer1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.layer2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)  
        self.layer3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)  

        self.layer4 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False, dropout=dropout)
        
        self.activation_function = nn.ELU(alpha=alpha)
        self.final_activation = nn.Tanh()


    def forward(self, input, edge_index):
        output = self.layer1(input, edge_index)
        output = self.activation_function(output)
        output = F.dropout(output, p=self.dropout_rate, training=self.training)
        
        output = self.layer2(output, edge_index)
        output = self.activation_function(output)
        output = F.dropout(output, p=self.dropout_rate, training=self.training)
        
        output = self.layer3(output, edge_index)
        output = self.activation_function(output)
        output = F.dropout(output, p=self.dropout_rate, training=self.training)
        
        output = self.layer4(output, edge_index)
        output = global_mean_pool(output, batch)
        output = self.final_activation(output)
        
        return output
    
    
input_dim = len(unique_nodes_bayern) + len(unique_nodes_opp) + 3
lr = 0.001

gat = GAT(alpha=0.005, input_dim = input_dim, hidden_dim = 50, output_dim = 1, num_heads = 5)
optimizer = torch.optim.SGD(gat.parameters(), lr=lr, weight_decay=1e-4)
loss_fn = torch.nn.MSELoss()
epochs_num = 100

for epoch in tqdm.tqdm(dataloader):
    
    epoch_loss = 0
    for idx in tqdm.tqdm(range(len(train_idx))):
        
        input = pyg_data_combined[idx].x
        edge_idx = pyg_data_combined[idx].edge_index
        label = pyg_data_combined[idx]['momentum']
        
        # Forward pass
        optimizer.zero_grad()
        
        output = gat(input, edge_idx)
        output = torch.tensor(output) if not isinstance(output, torch.Tensor) else output
        label = torch.tensor(label) if not isinstance(label, torch.Tensor) else label


        loss = loss_fn(output, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gat.parameters(), max_norm=1.0)

        for p in gat.parameters():
            p.data.add_(p.grad.data, alpha=-lr)
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_idx):.4f}") 
gat.eval()


y_pred  = []
y_true = []

with torch.no_grad():
    for idx in test_idx:
        output = gat(pyg_data_combined[idx].x, pyg_data_combined[idx].edge_index)
        y_pred.append(output.numpy())
        y_true.append(pyg_data_combined[idx].momentum)
        
y_true = np.array([np.array(y).flatten() for y in y_true])
y_pred = np.array([np.array(y).flatten() for y in y_pred])



mean_absolute_error(y_true, y_pred)
mean_absolute_percentage_error(y_true, y_pred)