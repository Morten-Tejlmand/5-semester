from statsbombpy import sb
import pandas as pd
import networkx as nx
import networkx as nx
from itertools import combinations
import multiprocessing
import numpy as np
from helper_functions import create_graphs, create_graphs_dict
multiprocessing.set_start_method('fork', force=True)

from itertools import combinations
from networkx.algorithms.isomorphism import DiGraphMatcher


matches = sb.matches(competition_id=9, season_id=281)
match_ids = matches['match_id'].values.tolist()

events = sb.competition_events(
    country="Germany",
    division= "1. Bundesliga",
    season="2023/2024",
    gender="male"
)
events = events[events['match_id'].isin(match_ids)]
df = events[events["team"]=="Bayer Leverkusen"]
df = df[df["possession_team"]=="Bayer Leverkusen"]

position_dict = {}
for id in events.match_id.unique():
    match_subset = events.loc[events['match_id'] == id]
    starting_11 = match_subset.loc[match_subset['type'] == 'Starting XI'].loc[match_subset['team'] == 'Bayer Leverkusen', 'tactics'].to_list()[0]
    starting_11_dict = {}

    
    #we make a dictionary for positions of players
    for member in starting_11['lineup']:
        player_id = member['player']['id']
        position_name = member['position']['name']
        starting_11_dict[player_id] = position_name
    
    position_dict[id] = {0 : starting_11_dict}

    match_subset = match_subset.loc[(match_subset['team'] == 'Bayer Leverkusen') & (match_subset['type'].isin(['Substitution', 'Tactical Shift']))]

    #sort the values like when we did the passing sequences
    match_subset = match_subset.sort_values(['period','timestamp'], ascending=[True, True])

    match_subset['pass_recipient_position'] = np.nan

    for index, row in match_subset.iterrows():
        #If substitution, we update the dictionary to include player
        if row['type'] == 'Substitution':
            match_dict = position_dict[id]
            latest_lineup = match_dict[list(match_dict.keys())[-1]]
            latest_lineup[row['substitution_replacement_id']] = row['position']
            position_dict[id][row['possession']] = latest_lineup
          
        #In case of a tactical shift, create a new position_dict
        if row['type'] == 'Tactical Shift':
            new_formation = row['tactics']
            lineup = {}
            for member in new_formation['lineup']:
                player_id = member['player']['id']
                position_name = member['position']['name']
                lineup[player_id] = position_name
            position_dict[id][row['possession']] = lineup


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
player_final_sequences =  sequences_filtered[sequences_filtered["pass_recipient"].notna()][["possession","player_id", "pass_recipient_id", "possession_id", "xg","timestamp","match_id"]]


player_final_sequences = player_final_sequences.sort_values(['possession_id', 'timestamp'], ascending=[True, True])
player_final_sequences = player_final_sequences[player_final_sequences['player_id'] != player_final_sequences['pass_recipient_id']]

player_final_sequences['sequence'] = player_final_sequences.groupby('possession_id').cumcount(ascending=False) + 1
player_final_sequences
#remove sequences with few passes if wanted
index_counts = player_final_sequences['possession_id'].value_counts()
player_final_sequences = player_final_sequences[player_final_sequences['possession_id'].isin(index_counts[index_counts > 5].index)]
possession_index = player_final_sequences["possession_id"].unique()


player_final_sequences['pass_recipient_position'] = np.nan

for index, row in player_final_sequences.iterrows():
    recipient = row['pass_recipient_id']
    passer = row['player_id']

    lineups = position_dict[row['match_id']]

    i = 0
    keys = list(lineups.keys())

    #Find out the current line up at the time of the event
    while i <= len(keys)-2 and row['possession'] > keys[i+1]:
        i += 1
    
    lineup = lineups[keys[i]]

    player_final_sequences.at[index, 'pass_recipient_id']  = lineup[recipient]
    player_final_sequences.at[index, 'player_id']  = lineup[passer]

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
    for graph in graph_db:
        for candidate in C:
            GM = DiGraphMatcher(graph, candidate)
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
            subgraph_edges = list(subgraph.edges(data=True))
            subgraph_sequence = sorted((edge[2]['sequence'] for edge in subgraph_edges))
            subgraph_sequence.sort()
            subgraph_nodes = sorted((node, tuple(attributes.items())) for node, attributes in subgraph.nodes(data=True))  # Include node attributes
            subgraph_nodes.sort()
            support_count = 0

            for reference_subgraph in F:
                reference_subgraph_edges = list(reference_subgraph.edges(data=True))
                reference_sequence = sorted((edge[2]['sequence'] for edge in reference_subgraph_edges))
                reference_sequence.sort()
                reference_nodes = sorted((node, tuple(attributes.items())) for node, attributes in reference_subgraph.nodes(data=True))
                reference_nodes.sort()
                
                if subgraph_sequence == reference_sequence and subgraph_nodes == reference_nodes:
                    support_count += 1 

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

possession_index, final_sequences = create_graphs(df)
graph_list, graphs_dict = create_graphs_dict(possession_index, final_sequences)


graph_list = [value["graph"] for value in graphs_dict.values()]

edge_matrix = [list(graph.edges(data=True)) for graph in graph_list]
GRAPH_DB = graph_list  
min_sup = 5
max_k = 7
frequent_subgraphs, patterns_df = apriori_graph_mining(5, edge_matrix, GRAPH_DB, 7)

graph_db = GRAPH_DB