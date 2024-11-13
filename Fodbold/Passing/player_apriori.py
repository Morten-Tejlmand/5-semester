from statsbombpy import sb
import pandas as pd
import networkx as nx
import networkx as nx
from itertools import combinations
import multiprocessing

from networkx.algorithms.isomorphism import DiGraphMatcher
multiprocessing.set_start_method('fork', force=True)



matches = sb.matches(competition_id=9, season_id=281)
match_ids = matches['match_id'].values.tolist()


#All events sorted for barca home games and possession 
#find ids for barca home matches - only home games because then locations can be compared between different games
events = sb.competition_events(
    country="Germany",
    division= "1. Bundesliga",
    season="2023/2024",
    gender="male"
)
events = events[events['match_id'].isin(match_ids)]
df = events[events["team"]=="Bayer Leverkusen"]
df = df[df["possession_team"]=="Bayer Leverkusen"]
#filter threshold for Xg:
df_xg = df[~df['shot_statsbomb_xg'].between(0, 0.05)]
#Events sorted in a specific order so each passing sequence is correctly sorted
sequences_sorted = df_xg.sort_values(['match_id', 'period','timestamp'], ascending=[True, True, True])
#make new ids because right now there is ids from 1 to x for each match but it repeats from 1 and up in every match so each possession id points to different matches 
# - i just put the possession id after match_id in the newly created id
sequences_sorted['possession_id'] = sequences_sorted['match_id'].astype(str) + sequences_sorted['possession'].astype(str)
sequences_sorted['possession_id'] = sequences_sorted['possession_id'].astype(int)
#get the ids of sequences which contain a shot (contain an xg value)
shot_sequences = sequences_sorted[sequences_sorted["shot_statsbomb_xg"].notna()]
shot_sequences_ids = shot_sequences["possession_id"].unique()
#filter for possession sequences which end with a shot
sequences_filtered = sequences_sorted[sequences_sorted['possession_id'].isin(shot_sequences_ids)]
#fill all rows with an xg for the corresponding sequence - right now there are many missing values in "shot_statsbomb_xg"
sequences_filtered['xg'] = sequences_filtered.groupby('possession_id')['shot_statsbomb_xg'].transform(lambda group: group.fillna(method='ffill').fillna(method='bfill'))
#now we dont need the shot event rows any more so remove them
sequences_filtered = sequences_filtered[sequences_filtered["type"]!="Shot"]
#filter the df to only include row with an id the of a pass recipient and we subset the columns
player_final_sequences =  sequences_filtered[sequences_filtered["pass_recipient"].notna()][["player_id", "pass_recipient_id", "possession_id", "xg","timestamp"]]
player_final_sequences
player_final_sequences = player_final_sequences.sort_values(['possession_id', 'timestamp'], ascending=[True, True])
player_final_sequences = player_final_sequences[player_final_sequences['player_id'] != player_final_sequences['pass_recipient_id']]
player_final_sequences['sequence'] = player_final_sequences.groupby('possession_id').cumcount(ascending=False) + 1
player_final_sequences
#remove sequences with few passes if wanted
index_counts = player_final_sequences['possession_id'].value_counts()
player_final_sequences = player_final_sequences[player_final_sequences['possession_id'].isin(index_counts[index_counts > 5].index)]
possession_index = player_final_sequences["possession_id"].unique()


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

    # min_sup = int(min_sup * len(edge_matrix))                           
    source = 0 
    for edge_list in edge_matrix:
        source +=1 
        for edge in edge_list:
            # Use only source and target nodes for counting
            edge_key = (edge[0], edge[1])
            items_counted[edge_key] = items_counted.get(edge_key, 0) + 1
            
            # Store the edge attributes
            edge_attributes[edge_key] = edge[2]
            edge_attributes[edge_key]['source'] = source  # Add the source ID to the edge attributes


    # Filter edges that meet the min_sup
    F = []
    for key, value in items_counted.items():
        if value >= min_sup:
            F.append(key)
    
    F_graphs = []
    for edge in F:
        g = nx.DiGraph()
        g.add_edge(edge[0], edge[1], **edge_attributes[edge])  # Add edge with original attributes
        g.graph['source'] = edge_attributes[edge].get("source")  # Add source graph ID to the graph metadata
        F_graphs.append(g)
    return F_graphs

def generate_candidates(F, k):
    candidates = set()

    # Iterate over all pairs of frequent subgraphs (F)
    for g1, g2 in combinations(F, 2):
        # Extract edges and sort them by their sequence number
        edges_g1 = sorted(g1.edges(data=True), key=lambda e: e[2].get('sequence'))
        edges_g2 = sorted(g2.edges(data=True), key=lambda e: e[2].get('sequence'))
        
        # Ensure graphs are from the same source graph by checking unique identifiers
        source_g1 = g1.graph.get("source")
        source_g2 = g2.graph.get("source")
        
        # Only combine graphs if they come from the same source graph
        if source_g1 == source_g2:
            last_sequence_g1 = edges_g1[-1][2]['sequence']  # Last event's sequence in g1
            first_sequence_g2 = edges_g2[0][2]['sequence']  # First event's sequence in g2
            
            # Combine graphs only if sequences are continuous
            if last_sequence_g1 + 1 == first_sequence_g2:
                union_graph = nx.compose(g1, g2)
                
                # Ensure that the union has the correct number of edges (k)
                if union_graph.number_of_edges() == k:
                    candidates.add(union_graph)

    return candidates

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
# Count the support for each candidate in the graph database
def count_support(C, graph_db):
    F_count = {}
    subgraph_support = {}

    for graph in graph_db:
        for candidate in C:
            # Use edge_match to compare both the structure and the sequence attribute
            GM = DiGraphMatcher(graph, candidate, edge_match=lambda x, y: x['sequence'] == y['sequence'])
            
            if GM.subgraph_is_isomorphic():  # Check for subgraph isomorphism with matching sequence
                F_count[candidate] = F_count.get(candidate, 0) + 1
                
                # Store support for individual nodes as well
                for subgraph in C:
                    GM_sub = DiGraphMatcher(graph, subgraph, edge_match=lambda x, y: x['sequence'] == y['sequence'])
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

    
    
graph_list = [value["graph"] for value in graphs_dict.values()]

graph_dict = {value["graph"]: value["xg"] for value in graphs_dict.values()}
    
edge_matrix = [list(graph.edges(data=True)) for graph in graph_dict]

GRAPH_DB = graph_list  # List of graphs in the database
min_sup = 0

