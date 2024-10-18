from statsbombpy import sb
import pandas as pd
import networkx as nx
import networkx as nx
from itertools import combinations
import multiprocessing

from networkx.algorithms.isomorphism import DiGraphMatcher
multiprocessing.set_start_method('fork', force=True)


matches = sb.matches(competition_id=11, season_id=90)
barca_home_matches = matches[matches["home_team"]=="Barcelona"]
match_ids = barca_home_matches['match_id'].values.tolist()


#All events sorted for barca home games and possession 
events = sb.competition_events(
    country="Spain",
    division= "La Liga",
    season="2020/2021",
    gender="male"
)
events = events[events['match_id'].isin(match_ids)]
df = events[events["team"]=="Barcelona"]
df = df[df["possession_team"]=="Barcelona"]

df_xg = df[~df['shot_statsbomb_xg'].between(0, 0.05)]
sequences_sorted = df_xg.sort_values(['match_id', 'period','timestamp'], ascending=[True, True, True])

sequences_sorted['possession_id'] = sequences_sorted['match_id'].astype(str) + sequences_sorted['possession'].astype(str)
sequences_sorted['possession_id'] = sequences_sorted['possession_id'].astype(int)

shot_sequences = sequences_sorted[sequences_sorted["shot_statsbomb_xg"].notna()]
shot_sequences_ids = shot_sequences["possession_id"].unique()

sequences_filtered = sequences_sorted[sequences_sorted['possession_id'].isin(shot_sequences_ids)]

sequences_filtered['xg'] = sequences_filtered.groupby('possession_id')['shot_statsbomb_xg'].transform(lambda group: group.fillna(method='ffill').fillna(method='bfill'))

goal_posession_index = sequences_filtered.loc[sequences_filtered["shot_outcome"]=="Goal", "possession_id"]

sequences_filtered = sequences_filtered[sequences_filtered["type"]!="Shot"]


player_final_sequences =  sequences_filtered[sequences_filtered["pass_recipient"].notna()][["player_id", "pass_recipient_id", "possession_id", "xg"]]
player_final_sequences['sequence'] = player_final_sequences.groupby('possession_id').cumcount(ascending=False) + 1



index_counts = player_final_sequences['possession_id'].value_counts()
player_final_sequences = player_final_sequences[player_final_sequences['possession_id'].isin(index_counts[index_counts > 5].index)]

possession_index = player_final_sequences["possession_id"].unique()
player_final_sequences



#iterate over possession ids and each row and append edges to a list for each graph and append that graph to a graphs dictionary (directed graph created with "nx.DiGraph(edges)")
#xg added as an attribute for each graph
graphs_dict = {}
for j in possession_index:
    edges = []
    for i in player_final_sequences.index:
        if j == player_final_sequences["possession_id"][i]:
            edge = (player_final_sequences["player_id"][i], player_final_sequences["pass_recipient_id"][i])
            edges.append(edge)
            if j not in graphs_dict:
                graphs_dict[j] = {"xg": player_final_sequences["xg"][i], "graph": None}
            else:
                graphs_dict[j]["xg"] = player_final_sequences["xg"][i]

    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    # Add sequence as an edge attribute
    for i in player_final_sequences.index:
        if j == player_final_sequences["possession_id"][i]:
            graph[player_final_sequences["player_id"][i]][player_final_sequences["pass_recipient_id"][i]]['sequence'] = player_final_sequences["sequence"][i]

    graphs_dict[j]["graph"] = graph


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
                # Output the edge along with its sequence attribute
                print(f"Edge {u} -> {v}, Attributes: {attr}")
        
        k += 1  # Move to the next size of subgraphs

    return frequent_total



graph_list = [value["graph"] for value in graphs_dict.values()]

edge_matrix = [list(graph.edges(data=True)) for graph in graph_list]
GRAPH_DB = graph_list  # List of graphs in the database
min_sup = 0


frequents = apriori_graph_mining(20, edge_matrix, GRAPH_DB, )
for subgraph in frequents:
    edges_with_sequences = [(u, v, attr['sequence']) for u, v, attr in subgraph.edges(data=True)]
    print(edges_with_sequences)