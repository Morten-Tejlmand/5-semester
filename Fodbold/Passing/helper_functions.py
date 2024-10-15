
from statsbombpy import sb
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx

HOME_TEAM = 'Barcelona'
COMPETITION_ID = 11
SEASON_ID = 90


def match_ids(events, home_team:str = HOME_TEAM,
              competition_id:int = COMPETITION_ID, season_id:int = SEASON_ID):
    """get all match ids for a specific team

    Args:
        events: season event 
        home_team (str, optional): team to analyze. Defaults to Barcelona.
        competition_id (int, optional):  Defaults to 11.
        season_id (int, optional):  Defaults to 90.
    """
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    barca_home_matches = matches[matches["home_team"]==home_team]
    match_ids = barca_home_matches['match_id'].values.tolist()

    events = events[events['match_id'].isin(match_ids)]
    df = events[events["team"]==home_team]
    df = df[df["possession_team"]==home_team]
    return df


def create_graphs(df, xG: float = 0.05, min_passes: int = 5):
    """
    creates a df ready to make each sequence into a graph

    Args:
        df (_type_): event_dataframe for a team
        xG (float, optional): threshold for xG Defaults to 0.05.
        min_passes (int, optional): threshold for min_passes Defaults to 5.

    Returns:
        possesion_index which is a list of int
        final_sequences which is a df of all the sequences
    """

    df_xg = df[~df['shot_statsbomb_xg'].between(0, xG)]

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

    #make end location for passes and carries (start location has no missing values)
    #combine the two types (pass and carry) into one location column (now we dont know if an edge is a pass or carry btw)
    sequences_filtered['end_location'] = sequences_filtered['location'].shift(-1)

    #now we dont need the shot event rows any more so remove them
    sequences_filtered = sequences_filtered[sequences_filtered["type"]!="Shot"]

    #assign x and y coordinates from location lists
    sequences_filtered["start_x"] = sequences_filtered["location"].str[0]
    sequences_filtered["start_y"] = sequences_filtered["location"].str[1]
    sequences_filtered["end_x"] = sequences_filtered["end_location"].str[0]
    sequences_filtered["end_y"] = sequences_filtered["end_location"].str[1]

    #reduce the number of possible x and y coordinates, essentially making the fields/nodes of the pitch larger
    sequences_filtered['start_node_x'] = round(sequences_filtered['start_x'] / 20)
    sequences_filtered['start_node_y'] = round(sequences_filtered['start_y'] / 20)
    sequences_filtered['end_node_x'] = round(sequences_filtered['end_x'] / 20)
    sequences_filtered['end_node_y'] = round(sequences_filtered['end_y'] / 20)

    #invert y values in second period
    sequences_filtered_p1 = sequences_filtered[sequences_filtered["period"]==1]
    sequences_filtered_p2 = sequences_filtered[sequences_filtered["period"]==2]
    sequences_filtered_p2["start_node_y"] = 4 - sequences_filtered_p2["start_node_y"] 
    sequences_filtered_p2["end_node_y"] = 4 - sequences_filtered_p2["end_node_y"] 
    sequences_filtered = pd.concat([sequences_filtered_p1,sequences_filtered_p2], axis=0, ignore_index=True)

    #combine the x and y coordinates
    sequences_filtered["start_node"] = sequences_filtered["start_node_x"] + sequences_filtered["start_node_y"] / 10
    sequences_filtered["end_node"] = sequences_filtered["end_node_x"] + sequences_filtered["end_node_y"] / 10
    
    
    
    # Add a sequence column by grouping for possession_id
    sequences_filtered = sequences_filtered.sort_values(['possession_id', 'timestamp'], ascending=[True, True])
    sequences_filtered = sequences_filtered[sequences_filtered['start_node'] != sequences_filtered['end_node']]
    sequences_filtered['sequence'] = sequences_filtered.groupby('possession_id').cumcount(ascending=False) + 1

    # Filter out rows where start_node and end_node are the same
    sequences_filtered = sequences_filtered[["start_node","end_node","xg","possession_id", "sequence"]]

    #remove edges between the same node if wanted
    final_sequences = sequences_filtered[sequences_filtered['start_node'] != sequences_filtered['end_node']]

    #remove rows with missing values - there are only a few
    final_sequences.dropna(inplace=True)

    #remove sequences with few passes if wanted
    index_counts = final_sequences['possession_id'].value_counts()
    final_sequences = final_sequences[final_sequences['possession_id'].isin(index_counts[index_counts > min_passes].index)]

    #get all the unique possession ids for iteration
    possession_index = final_sequences["possession_id"].unique()
    return possession_index, final_sequences


def create_graphs_dict(possession_index, final_sequences):
    """
    create a graph for each possession id

    Args:
        possession_index 
        final_sequences 

    Returns:
        graph_list a list contating only all graphs
        graphs_dict a dict with possesion id and the corresponding graph and min xg
    """
    #iterate over possession ids and each row and append edges to a list for each graph and append that graph to a graphs dictionary (directed graph created with "nx.DiGraph(edges)")
    # xg added as an attribute for each graph
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

    graph_list = [value["graph"] for value in graphs_dict.values()]
    return graph_list, graphs_dict

def plot_graphs(graphs_dict,possession_index):
    """plot a graph with edge attributes

        Args:
            graphs_dict:
            possession_index: a specific possession index

        Returns:
            a graph
        """
    positions = {}
    for node in graphs_dict[possession_index]["graph"].nodes():
        x = int(node)  # Integer part represents the x position
        y = node - x   # Decimal part represents the y position
        positions[node] = (x, y)

    nx.draw_networkx(graphs_dict[possession_index]["graph"], pos=positions)

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(graphs_dict[possession_index]["graph"], 'sequence')
    nx.draw_networkx_edge_labels(graphs_dict[possession_index]["graph"], pos=positions, edge_labels=edge_labels)

    plt.show()
    return positions

