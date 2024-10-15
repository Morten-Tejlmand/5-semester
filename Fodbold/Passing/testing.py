import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.isomorphism import GraphMatcher, DiGraphMatcher
from itertools import combinations, chain
from mlxtend.frequent_patterns import apriori, association_rules
import multiprocessing


import pandas as pd
from helper_functions import(match_ids,
                             create_graphs,
                             create_graphs_dict,
                             plot_graphs)
# Ensure that the 'fork' method is used explicitly for multiprocessing
multiprocessing.set_start_method('fork', force=True)
from statsbombpy import sb

events = sb.competition_events(
        country="Germany",
        division="1. Bundesliga",
        season="2023/2024",
        gender="male"
)
    
df = match_ids(events, "Bayer Leverkusen", season_id=281, competition_id=9)
possesion, final_sequence = create_graphs(df)
graph_list, graph_dict = create_graphs_dict(possesion, final_sequence)


graph_list_sample = graph_list

# Create a list of edges from the sampled graph_list
edge_matrix = [list(graph.edges()) for graph in graph_list_sample]
GRAPH_DB = graph_list_sample  # List of graphs in the database
min_sup = 0


edge_matrix = [list(graph.edges(data=True)) for graph in graph_list_sample]

def frequent_singletons(min_sup, edge_matrix):
    items_counted = {}
    
    for edge_list in edge_matrix:
        for edge in edge_list:
            # Use both source, target nodes and sequence attribute for counting
            edge_key = (edge[0], edge[1], edge[2].get('sequence', None))
            items_counted[edge_key] = items_counted.get(edge_key, 0) + 1
  
    # Filter edges that meet the min_sup
    F = [key for key, value in items_counted.items() if value >= min_sup]
    
    F_graphs = []
    for edge in F:
        g = nx.DiGraph()
        g.add_edge(edge[0], edge[1])  # Add edge without sequence for the subgraph
        F_graphs.append(g)
    
    return F_graphs



# Step 1: Find frequent singletons (edges)
F = frequent_singletons(0, edge_matrix)
k = 2

# Generate candidates of size k subgraphs from frequent subgraphs
def generate_candidates(F, k):
    candidates = []
    for i, g1 in enumerate(F):
        for g2 in F[i + 1:]:
            # Sort edges of g1 by 'sequence' attribute
            sorted_edges_g1 = sorted(g1.edges(data=True), key=lambda x: x[2].get('sequence', float('inf')))
            sorted_edges_g2 = sorted(g2.edges(data=True), key=lambda x: x[2].get('sequence', float('inf')))
            
            # Get the last edge of g1 and the first edge of g2
            last_edge_g1 = sorted_edges_g1[-1]  # Get the last edge (highest sequence)
            first_edge_g2 = sorted_edges_g2[0]  # Get the first edge (lowest sequence)

            # Ensure that the graphs can only be merged if they follow the correct time order
            if last_edge_g1[2].get('sequence', 0) - first_edge_g2[2].get('sequence', 0) == -1:
                union_graph = nx.compose(g1, g2)

                # Ensure that the union has the correct number of nodes
                if len(union_graph.nodes) == k + 1:
                    candidates.append(union_graph)

    return candidates

# Count the support for each candidate in the graph database
def count_support(C, graph_db):
    F_count = {}
    for graph in graph_db:
        for candidate in C:
            GM = DiGraphMatcher(graph, candidate)
            if GM.subgraph_is_isomorphic():  # Check for subgraph isomorphism
                F_count[candidate] = F_count.get(candidate, 0) + 1
    return F_count

# Filter frequent candidates based on minimum support
def filter_frequent(F_count, min_sup):
    return [key for key, value in F_count.items() if value >= min_sup]

# Main function to run the apriori graph mining algorithm
def apriori_graph_mining(min_sup, edge_matrix, graph_db, max_k):
    frequent_total = []
    
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
        F_count = count_support(C, graph_db)
        
        # Step 4: Filter out frequent candidates that meet the minimum support
        F = filter_frequent(F_count, min_sup)
        
        if not F:  # If no frequent candidates are found, stop the algorithm
            print(f"No frequent subgraphs found for size {k}. Terminating.")
            break
        
        # Add frequent items to the total list
        frequent_total.extend(F)
        
        print(f"Frequent subgraphs of size {k}:")
        for subgraph in F:
            for u, v, attr in subgraph.edges(data=True):
                print(f"Edge {u} -> {v}, Attributes: {attr}")
        
        k += 1  # Move to the next size of subgraphs

    return frequent_total


frequents = apriori_graph_mining(1, edge_matrix, graph_list_sample, 1000)
for pattern in frequents:
    print(pattern.edges())

for i, graph in enumerate(F):
    print(f"\nGraph {i+1} with sequence attributes:")
    
    # Print Edge Attributes
    print("\nEdges and their sequence attributes:")
    for u, v, attr in graph.edges(data=True):
        print(f"Edge {u} -> {v}, Attributes: {attr}")


