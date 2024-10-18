import networkx as nx
from itertools import combinations
from networkx.algorithms.isomorphism import DiGraphMatcher
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

from statsbombpy import sb
import pandas as pd
from helper_functions import(match_ids,
                             create_graphs,
                             create_graphs_dict,
                             )


def frequent_singletons(min_sup, edge_matrix):
    items_counted = {}
    edge_attributes = {}

    for edge_list in edge_matrix:
        for edge in edge_list:
            # Use only source and target nodes for counting
            edge_key = (edge[0], edge[1])
            items_counted[edge_key] = items_counted.get(edge_key, 0) + 1
            
            # Store the edge attributes
            if edge_key not in edge_attributes:
                edge_attributes[edge_key] = edge[2]

    # Filter edges that meet the min_sup
    F = [key for key, value in items_counted.items() if value >= min_sup]

    F_graphs = []
    for edge in F:
        g = nx.DiGraph()
        g.add_edge(edge[0], edge[1], **edge_attributes[edge])  # Add edge with original attributes
        F_graphs.append(g)
    return F_graphs

# Generate candidates of size k subgraphs from frequent subgraphs
def generate_candidates(F, k):
    candidates = set()
    
    # Iterate over all pairs of frequent subgraphs (F)
    for g1, g2 in combinations(F, 2):

        edges_g1 = sorted(g1.edges(data=True), key=lambda e: e[2].get('sequence', 0))
        edges_g2 = sorted(g2.edges(data=True), key=lambda e: e[2].get('sequence', 0))
        edges_g1 = edges_g1[0]
        edges_g2 = edges_g2[0]
        
        if edges_g1[2].get('sequence') > edges_g2[2].get('sequence') or edges_g1[2].get('sequence') < edges_g2[2].get('sequence'):
            union_graph = nx.compose(g1, g2)
            
            # Ensure that the union has the correct number of edges
            if union_graph.number_of_edges() == k:  # Check edge size instead of node size
                candidates.add(union_graph)
    return candidates

# Count the support for each candidate in the graph database
def count_support(C, graph_db):
    F_count = {}
    subgraph_support = {}

    for graph in graph_db:
        for candidate in C:
            GM = DiGraphMatcher(graph, candidate)
            if GM.subgraph_is_isomorphic():  # Check for subgraph isomorphism
                F_count[candidate] = F_count.get(candidate, 0) + 1
                
                # Store support for individual nodes as well
                for subgraph in C:
                    GM_sub = DiGraphMatcher(graph, subgraph)
                    if GM_sub.subgraph_is_isomorphic():
                        subgraph_support[subgraph] = subgraph_support.get(subgraph, 0) + 1
    
    return F_count, subgraph_support


# Filter frequent candidates based on minimum support
def filter_frequent(F_count, subgraph_support, min_sup, graph_db_size):
    frequent_graphs = []
    stats = {}

    for candidate, support_AB in F_count.items():
        if support_AB >= min_sup:
            # Calculate support for antecedent (A)
            support_A = subgraph_support.get(candidate, 0)
            
            # Confidence: P(B|A) = Support(A,B) / Support(A)
            confidence = support_AB / support_A if support_A > 0 else 0

            # Lift: Lift(A,B) = P(A,B) / (P(A) * P(B))
            support_B = F_count.get(candidate, 0)  # Simplified for single graph case
            lift = (support_AB / graph_db_size) / ((support_A / graph_db_size) * (support_B / graph_db_size)) if support_A > 0 and support_B > 0 else 0
            
            # Leverage: Leverage(A,B) = Support(A,B) - (Support(A) * Support(B))
            leverage = (support_AB / graph_db_size) - ((support_A / graph_db_size) * (support_B / graph_db_size))

            # Conviction: Conviction(A,B) = (1 - Support(B)) / (1 - Confidence(A->B))
            conviction = (1 - support_B / graph_db_size) / (1 - confidence) if (1 - confidence) > 0 else 0

            # Store the statistics
            stats[candidate] = {
                'support': support_AB,
                'confidence': confidence,
                'lift': lift,
                'leverage': leverage,
                'conviction': conviction
            }

            frequent_graphs.append(candidate)
    
    return frequent_graphs, stats
def edge_attr_match(attr1, attr2):
    return attr1 == attr2  # Check if both edge attribute dictionaries are identical

def remove_duplicates(frequent_total):
    unique_graphs = []

    for graph in frequent_total:
        is_duplicate = False
        
        # Check against all graphs already in the unique list
        for unique_graph in unique_graphs:
            # Use DiGraphMatcher with edge attribute comparison
            GM = DiGraphMatcher(graph, unique_graph, edge_match=edge_attr_match)
            if GM.is_isomorphic():  # Check if the graphs are isomorphic including edge attributes
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_graphs.append(graph)

    return unique_graphs
# Main function to run the apriori graph mining algorithm
def apriori_graph_mining(min_sup, edge_matrix, graph_db, max_k):
    frequent_total = []
    stats_total = {}
    
    # Step 1: Find frequent singletons (edges)
    F = frequent_singletons(min_sup, edge_matrix)
    
    # Add initial frequent items to the total list
    frequent_total.extend(F)
    
    k = 2  # Start with size-2 subgraphs
    while k <= max_k:
        print(f"\nIteration {k}:")
        
        # Step 2: Generate candidate subgraphs of size k
        C = generate_candidates(F, k)
        
        # Step 3: Count support for each candidate in the graph database
        F_count, subgraph_support = count_support(C, graph_db)
        
        # Step 4: Filter out frequent candidates that meet the minimum support
        F, stats = filter_frequent(F_count, subgraph_support, min_sup, len(graph_db))
        
        if not F:  # If no frequent candidates are found, stop the algorithm
            print(f"No frequent subgraphs found for size {k}. Terminating.")
            break
        
        # Add frequent items and statistics to the total list
        frequent_total.extend(F)
        stats_total.update(stats)
        
        print(f"Frequent subgraphs of size {k}:")
        for subgraph in F:
            for u, v, attr in subgraph.edges(data=True):
                print(f"Edge {u} -> {v}, Attributes: {attr}")
        
        k += 1  # Move to the next size of subgraphs

    frequent_total = remove_duplicates(frequent_total)

    return frequent_total, stats_total





k =2
events = sb.competition_events(
    country="Germany",
    division= "1. Bundesliga",
    season="2023/2024",
    gender="male"
)

df = match_ids(events, "Bayer Leverkusen", season_id=281, competition_id=9)
possesion, final_sequence = create_graphs(df)
graph_list, graph_dict = create_graphs_dict(possesion, final_sequence)


graph_list_sample = graph_list

# Create a list of edges from the sampled graph_list
edge_matrix = [list(graph.edges(data=True)) for graph in graph_list]
GRAPH_DB = graph_list_sample  # List of graphs in the database
min_sup = 0
xG = 0.5
min_passes = 5

frequent_subgraphs, stats_total = apriori_graph_mining(35, edge_matrix, GRAPH_DB, 40)

# Iterate over the frequent subgraphs
for subgraph in frequent_subgraphs:
    # Get the edges of the subgraph
    edges = list(subgraph.edges(data=True))
    
    # Get the corresponding statistics for this subgraph
    stats = stats_total.get(subgraph, {})
    
    # Print the edges of the subgraph
    print("Subgraph Edges:")
    for u, v, attr in edges:
        print(f"Edge {u} -> {v}, Attributes: {attr}")
    
    # Print the associated statistics
    print("Statistics:")
    print(f"Support: {stats.get('support', 'N/A')}")
    print(f"Confidence: {stats.get('confidence', 'N/A')}")
    print(f"Lift: {stats.get('lift', 'N/A')}")
    print(f"Leverage: {stats.get('leverage', 'N/A')}")
    print(f"Conviction: {stats.get('conviction', 'N/A')}")
    print("\n" + "-"*40 + "\n")  # Separator between subgraphs

subgraph_data = []

# Iterate over the frequent subgraphs
for subgraph in frequent_subgraphs:
    # Get the edges of the subgraph
    edges = list(subgraph.edges(data=True))
    
    # Get the corresponding statistics for this subgraph
    stats = stats_total.get(subgraph, {})
    
    # Convert the edges and stats into a dictionary
    subgraph_info = {
        'edges': [(u, v, attr) for u, v, attr in edges],
        'support': stats.get('support', 'N/A'),
        'confidence': stats.get('confidence', 'N/A'),
        'lift': stats.get('lift', 'N/A'),
        'leverage': stats.get('leverage', 'N/A'),
        'conviction': stats.get('conviction', 'N/A')
    }
    
    # Append the subgraph data to the list
    subgraph_data.append(subgraph_info)

# Create a DataFrame from the collected data
df_subgraphs = pd.DataFrame(subgraph_data)

# Display the DataFrame
print(df_subgraphs)

# Optionally, save to a CSV file
df_subgraphs.to_csv('frequent_subgraphs.csv', index=False)