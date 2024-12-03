from statsbombpy import sb
import pandas as pd
import networkx as nx
import networkx as nx
from itertools import combinations
import multiprocessing
import numpy as np

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

df_xg = df[~df['shot_statsbomb_xg'].between(0, 0.05)]
sequences_sorted = df_xg.sort_values(['match_id', 'period','timestamp'], ascending=[True, True, True])

sequences_sorted['possession_id'] = sequences_sorted['match_id'].astype(str) + sequences_sorted['possession'].astype(str)
sequences_sorted['possession_id'] = sequences_sorted['possession_id'].astype(int)

shot_sequences = sequences_sorted[sequences_sorted["shot_statsbomb_xg"].notna()]
shot_sequences_ids = shot_sequences["possession_id"].unique()

sequences_filtered = sequences_sorted[sequences_sorted['possession_id'].isin(shot_sequences_ids)]
sequences_filtered['xg'] = sequences_filtered.groupby('possession_id')['shot_statsbomb_xg'].transform(lambda group: group.fillna(method='ffill').fillna(method='bfill'))

sequences_filtered['end_location'] = sequences_filtered['location'].shift(-1)
sequences_filtered = sequences_filtered[sequences_filtered["type"]!="Shot"]

sequences_filtered["start_x"] = sequences_filtered["location"].str[0]
sequences_filtered["start_y"] = sequences_filtered["location"].str[1]
sequences_filtered["end_x"] = sequences_filtered["end_location"].str[0]
sequences_filtered["end_y"] = sequences_filtered["end_location"].str[1]


sequences_filtered['start_node_x'] = round(sequences_filtered['start_x'] / 20)
sequences_filtered['start_node_y'] = round(sequences_filtered['start_y'] / 20)
sequences_filtered['end_node_x'] = round(sequences_filtered['end_x'] / 20)
sequences_filtered['end_node_y'] = round(sequences_filtered['end_y'] / 20)

sequences_filtered_p1 = sequences_filtered[sequences_filtered["period"]==1]
sequences_filtered_p2 = sequences_filtered[sequences_filtered["period"]==2]
sequences_filtered_p2["start_node_y"] = 4 - sequences_filtered_p2["start_node_y"] 
sequences_filtered_p2["end_node_y"] = 4 - sequences_filtered_p2["end_node_y"] 

sequences_filtered = pd.concat([sequences_filtered_p1,sequences_filtered_p2], axis=0, ignore_index=True)

sequences_filtered["start_node"] = sequences_filtered["start_node_x"] + sequences_filtered["start_node_y"] / 10
sequences_filtered["end_node"] = sequences_filtered["end_node_x"] + sequences_filtered["end_node_y"] / 10


sequences_filtered['time'] = sequences_filtered['minute'] *60 + sequences_filtered['second']
sequences_filtered = sequences_filtered.sort_values(['possession_id', 'timestamp'], ascending=[True, True])

sequences_filtered = sequences_filtered[sequences_filtered['start_node'] != sequences_filtered['end_node']]
sequences_filtered['sequence'] = sequences_filtered.groupby('possession_id').cumcount(ascending=False) + 1

example = sequences_filtered[["start_node","end_node","xg","possession_id","sequence","minute","second",'timestamp']]
example = example.sort_values(['possession_id','sequence'], ascending=[True, True])

final_sequences = sequences_filtered[sequences_filtered['start_node'] != sequences_filtered['end_node']]


final_sequences.dropna(subset=['start_node', 'end_node'], inplace=True)

index_counts = final_sequences['possession_id'].value_counts()
final_sequences = final_sequences[final_sequences['possession_id'].isin(index_counts[index_counts > 5].index)]


possession_index = final_sequences["possession_id"].unique()


graphs_dict = {}
for j in possession_index:
    edges = []
    for i in final_sequences.index:
        if j == final_sequences["possession_id"][i]:
            edge = (final_sequences["start_node"][i], final_sequences["end_node"][i])
            edges.append(edge)
            if j not in graphs_dict:
                graphs_dict[j] = {"xg": final_sequences["xg"][i], "graph": None}
            else:
                graphs_dict[j]["xg"] = final_sequences["xg"][i]

    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    # Add sequence as an edge attribute
    for i in final_sequences.index:
        if j == final_sequences["possession_id"][i]:
            graph[final_sequences["start_node"][i]][final_sequences["end_node"][i]]['sequence'] = final_sequences["sequence"][i]

    graphs_dict[j]["graph"] = graph



def frequent_singletons(min_sup, edge_matrix, total_graphs):
    items_counted = {}
    edge_attributes = {}

    for idx, edge_list in enumerate(edge_matrix):
        for edge in edge_list:

            edge_key = (edge[0], edge[1])
            items_counted[edge_key] = items_counted.get(edge_key, 0) + 1
            
            if edge_key not in edge_attributes:
                edge_attributes[edge_key] = {**edge[2], 'source': idx}

    F = []
    for key, value in items_counted.items():
        support_percentage = (value / total_graphs) * 100  
        if support_percentage >= min_sup:
            F.append(key)

    
    F_graphs = []
    for edge in F:
        g = nx.DiGraph()

        g.add_edge(edge[0], edge[1], **edge_attributes[edge])
        F_graphs.append(g)
    
    return F_graphs

def generate_candidates(F, k):
    
    candidates = set()

    for g1, g2 in combinations(F, 2):

        edge_set_g1 = tuple(sorted(g1.edges(data=True)))
        edge_set_g2 = tuple(sorted(g2.edges(data=True)))
        
        source_g1 = next((edge[2].get('source') for edge in edge_set_g1 if 'source' in edge[2]), None)
        source_g2 = next((edge[2].get('source') for edge in edge_set_g2 if 'source' in edge[2]), None)

        if source_g1 != source_g2:
            continue

        edges_g1 = sorted(g1.edges(data=True), key=lambda e: e[2].get('sequence', 0))
        edges_g2 = sorted(g2.edges(data=True), key=lambda e: e[2].get('sequence', 0))
        
        if (abs(edges_g1[0][2].get('sequence') - edges_g2[0][2].get('sequence')) == 1
            or abs(edges_g2[0][2].get('sequence') - edges_g1[0][2].get('sequence')) == 1
            ):
            
            union_graph = nx.compose(g1, g2)
            
            if union_graph.number_of_edges() == k:  
                candidates.add(union_graph)
    
    return candidates


def count_support(C, graph_db):
    F_count = {}
    
    def edge_match(attr1, attr2):
        return attr1.get('sequence') == attr2.get('sequence')
    
    for graph in graph_db:
        for candidate in C:
            GM = DiGraphMatcher(graph, candidate, edge_match=edge_match)
            if GM.subgraph_is_isomorphic():  
                F_count[candidate] = F_count.get(candidate, 0) + 1
                
    return F_count

# Filter frequent candidates based on minimum support (in percentage)
def filter_frequent(F_count, min_sup, total_graphs):
    return [
        key for key, value in F_count.items()
        if (value / total_graphs) * 100 >= min_sup
    ]
# main function
def apriori_graph_mining(min_sup, edge_matrix, graph_db, max_k):
    frequent_total = []
    results = []
    total_graphs = len(graph_db)  

    F = frequent_singletons(min_sup, edge_matrix, total_graphs)

    frequent_total.extend(F)

    k = 2  
    while k <= max_k:
        print(f"\nIteration {k}:")

        
        C = generate_candidates(F, k)
        F_count = count_support(C, graph_db)
        F = filter_frequent(F_count, min_sup, total_graphs)

        if not F: 
            break

        frequent_total.extend(F)

        for subgraph in F:
            support_count = F_count[subgraph]
            support_percentage = (support_count / total_graphs) * 100

            edges_with_attrs = [
                (u, v, attr) for u, v, attr in subgraph.edges(data=True)
            ]
            results.append(
                {
                    "k": k,
                    "edges": edges_with_attrs,
                    "support_percentage": support_percentage, 
                }
            )

        k += 1

    results_df = pd.DataFrame(results)
    return frequent_total, results_df

graph_list = [value["graph"] for value in graphs_dict.values()]

edge_matrix = [list(graph.edges(data=True)) for graph in graph_list]
GRAPH_DB = graph_list  
graph_db = GRAPH_DB
min_sup = 5
frequent_subgraphs, patterns_df = apriori_graph_mining(5, edge_matrix, GRAPH_DB, 7)

