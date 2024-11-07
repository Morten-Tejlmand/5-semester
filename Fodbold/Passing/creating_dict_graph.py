	
from statsbombpy import sb
import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
import multiprocessing



multiprocessing.set_start_method('fork', force=True)

events = sb.competition_events(
    country="Germany",
    division= "1. Bundesliga",
    season="2023/2024",
    gender="male"
)

# Initialize a list to store processed data for each half with sequence information
df_lst = []
half_sequence = 0  # Initialize sequence counter for halves
match_counter = 0  # Initialize counter for matches
#make a list to hold a df of the events of a match
df_dict = {}

for id in events.match_id.unique():
    match_counter += 1
    #find one match for one team
    match_subset = events.loc[events['match_id'] == id]

    #we identify the starting 11
    starting_11 = match_subset.loc[match_subset['type'] == 'Starting XI'].loc[match_subset['team'] == 'Bayer Leverkusen', 'tactics'].to_list()[0]
    position_dict = {}
    
    #we make a dictionary for positions of players
    for member in starting_11['lineup']:
        player_id = int(member['player']['id'])
        position_name = member['position']['name']
        position_dict[player_id] = position_name


    #include only passes, shots, substitution or tactical shifts
    match_subset = match_subset.loc[(match_subset['type'].isin(['Shot','Pass','Substitution', 'Tactical Shift']))]

    #We include rows with Leverkusen as possession team or with type substitution or tactical shift
    match_subset = match_subset.loc[(match_subset['possession_team'] == 'Bayer Leverkusen') | (match_subset['type'].isin(['Substitution', 'Tactical Shift']))]

    #sort the values like when we did the passing sequences
    match_subset = match_subset.sort_values(['period','timestamp'], ascending=[True, True])

    match_subset['pass_recipient_position'] = np.nan

    for index, row in match_subset.iterrows():
        #If substitution, we update the dictionary to include player
        if row['type'] == 'Substitution' and row['team'] == 'Bayer Leverkusen':
            position_dict[row['substitution_replacement_id']] = row['position']

        #In case of a tactical shift, create a new position_dict
        if row['type'] == 'Tactical Shift' and row['team'] == 'Bayer Leverkusen':
            lineup = row['tactics']
            position_dict = {}
            for member in lineup['lineup']:
                player_id = int(member['player']['id'])
                position_name = member['position']['name']
                position_dict[player_id] = position_name

        elif row['pass_recipient_id'] in list(position_dict.keys()):
            #if the player is in our dictionary, we assign his position - else he may be a player of the opponent, and we ignore him
            match_subset.at[index, 'pass_recipient_position'] = position_dict[int(row['pass_recipient_id'])]
            
    #den næste linje kan eventuelt kommenteres ud, og så kan vi bare fjerne na i stedet, hvis vi ikke vil have de ekstra nodes, jeg foreslår :D
    #add shot as a pass recipient as well
    match_subset.loc[match_subset['type'] =='Shot', 'pass_recipient_position'] = match_subset.apply(lambda x: np.where(pd.isna(x['pass_recipient_position']), x['shot_outcome'], x['pass_recipient_position']), axis=1)

    #if a does not have a pass_recipient_position yet, we drop it
    match_subset.dropna(subset=['pass_recipient_position'], inplace=True)
    match_subset['match_sequence'] = match_counter  
    # Process each half separately
    half_sequence = 0  # Initialize sequence counter for halves
    for period in [1, 2]:  # Assuming '1' is the first half and '2' is the second half
        half_data = match_subset[match_subset['period'] == period].copy()
        if half_data.empty:
            continue  # Skip if no data for this half

        # Increment half sequence for each half processed
        half_sequence += 1
        half_data['half_sequence'] = half_sequence  # Add half_sequence column for tracking

        # Append the processed half to the result list
        df_lst.append(half_data)
    df_dict[id] = match_subset  

match_graph_dict = {}

# Iterate through the DataFrames in df_lst, which represent each half of each match
for match in df_lst:
    half_sequence = match['half_sequence'].iloc[0]  # Get the unique half sequence identifier
    match_sequence = match['match_sequence'].iloc[0]  # Get the unique match sequence identifier
    match_id = match['match_id'].iloc[0]

    # Use match_counter to identify each match sequence uniquely
    key = (match_sequence, match_id, half_sequence)

    # Initialize directed graph for this half
    graph = nx.DiGraph()
    edges = []

    # Add nodes for each unique position involved in passing events
    for node in pd.concat([match['position'], match['pass_recipient_position']], axis=0):
        if str(node) not in graph.nodes:
            graph.add_node(str(node))  # Add position as a node

    # Add edges for each pass, creating pairs of (position, pass recipient position)
    for passing in match[['position', 'pass_recipient_position']].itertuples():
        edge = (str(passing.position), str(passing.pass_recipient_position))
        edges.append(edge)

    # Count edge frequencies (passes between positions)
    edges_counter = Counter(edges)
    edge_and_count = [(edge[0], edge[1], edges_counter[edge]) for edge in edges]

    # Add weighted edges to the graph based on the count of each passing connection
    graph.add_weighted_edges_from(edge_and_count)

    # Store the completed graph in the dictionary with a tuple key (match_counter, match_id, half_sequence)
    match_graph_dict[key] = graph
    

# save this dict as a csv
graph_data = []

# Iterate through each graph in the dictionary
for (match_sequence, match_id, half_sequence), graph in match_graph_dict.items():
    # For each edge in the graph, get the start node, end node, and weight (frequency)
    for u, v, data in graph.edges(data=True):
        weight = data['weight']  # The frequency of passes
        
        # Append row to graph_data with match and half details
        graph_data.append({
            'match_sequence': match_sequence,
            'match_id': match_id,
            'half_sequence': half_sequence,
            'start_position': u,
            'end_position': v,
            'pass_frequency': weight
        })

# Convert list of rows to a DataFrame
graph_df = pd.DataFrame(graph_data)




graph_df.to_csv('/Users/morten/Desktop/p5 kode/5-semester/Fodbold/match_graph_data.csv', index=False)
