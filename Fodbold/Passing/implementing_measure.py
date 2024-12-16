
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx



df = pd.read_csv('/Users/MathildeStouby/Desktop/P5 GitHub/5-semester/Fodbold/match_graph_data.csv')

to_drop = df.loc[df['end_position'].isin(['Off T', 'Saved', 'Blocked', 'Goal', 'Wayward','Post', 'Saved to Post', 'Saved Off Target'])].index

df.drop(to_drop, inplace=True)
df.drop(columns=['half_sequence'], inplace=True)
df = df.groupby(['start_position', 'end_position', 'match_id'])['pass_frequency'].sum().reset_index()

    
match_dict_limit3 = {}
match_dict = {}
for match_id, group in df.groupby('match_id'):
    G3 = nx.DiGraph()
    G = nx.DiGraph()
    # Add edges with weights
    for _, row in group.iterrows():
        G.add_edge(row['start_position'], row['end_position'], weight=row['pass_frequency'])
        if row['pass_frequency'] > 3:
            G3.add_edge(row['start_position'], row['end_position'], weight=row['pass_frequency'])
    # Store the graph in the dictionary with match_id as the key
    match_dict_limit3[match_id] = G3
    match_dict[match_id] = G


# pagerank
position = ['Center Forward',
 'Right Center Back',
 'Right Defensive Midfield',
 'Center Back',
 'Left Attacking Midfield',
 'Left Wing Back',
 'Right Attacking Midfield',
 'Right Wing Back',
 'Left Defensive Midfield',
 'Goalkeeper',
 'Left Center Back']
oder = ['Goalkeeper', 'Center Back', 'Right Center Back', 'Left Center Back', 'Right Wing Back', 'Left Wing Back', 'Right Defensive Midfield', 'Left Defensive Midfield', 'Right Attacking Midfield', 'Left Attacking Midfield', 'Center Forward']


df_pagerank = pd.DataFrame(columns=position)
for key, value in match_dict.items():
    pagerank = nx.pagerank(value, weight='weight')
    df_pagerank = pd.concat([df_pagerank, pd.DataFrame(pagerank, index=[key])], ignore_index=True)

na_count = df_pagerank.isna().sum()[:11]
df_subset = df_pagerank[na_count.index]

# make a box plot for each column

position = ['Goalkeeper', 'Right Defensive Midfield', 'Left Defensive Midfield'] 
default_color = "lightgrey"
highlight_color = "cornflowerblue"
custom_palette = {column: (highlight_color if column in position else default_color) for column in df_subset.columns}

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_subset, orient='h', palette=custom_palette, showfliers=False, order=oder)
sns.stripplot(data=df_subset, alpha=0.7, color='black', orient='h', size=4, order=oder)
plt.savefig('boxplot_pagerank.pdf', bbox_inches='tight')



df_closeness = pd.DataFrame(columns=position)
for key, value in match_dict_limit3.items():
    closeness = nx.closeness_centrality(value)
    df_closeness = pd.concat([df_closeness, pd.DataFrame(closeness, index=[key])], ignore_index=True)

na_count = df_closeness.isna().sum().sort_values(ascending=True)[:11]
df_close_sub  = df_closeness[na_count.index]

position = ['Left Defensive Midfield', 'Center Forward', 'Goalkeeper'] 
default_color = "lightgrey"
highlight_color = "cornflowerblue"
custom_palette = {column: (highlight_color if column in position else default_color) for column in df_close_sub.columns}

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_close_sub, orient='h', showfliers=False, palette=custom_palette, order=oder)
sns.stripplot(data=df_close_sub, alpha=0.7, color='black', orient='h', size=4, order=oder)
plt.savefig('boxplot_closness.pdf', bbox_inches='tight')



df_betweness = pd.DataFrame(columns=position)
for key, value in match_dict_limit3.items():
    closeness = nx.betweenness_centrality(value)
    df_betweness = pd.concat([df_betweness, pd.DataFrame(closeness, index=[key])], ignore_index=True)

na_count = df_betweness.isna().sum().sort_values(ascending=True)[:11]
df_between_sub  = df_betweness[na_count.index]

position = ['Left Center Back', 'Left Defensive Midfield', 'Center Forward'] 
default_color = "lightgrey"
highlight_color = "cornflowerblue"
custom_palette = {column: (highlight_color if column in position else default_color) for column in df_between_sub.columns}

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_between_sub, orient='h', showfliers=False, palette=custom_palette, order=oder)
sns.stripplot(data=df_between_sub, alpha=0.7, color='black', orient='h', size=4, order=oder)
plt.savefig('boxplot_between.pdf', bbox_inches='tight')



# Create a simple graph to display differences in centrality measures
G = nx.Graph()
edges = [
    (1, 2), (1, 3), (2, 4), (3, 4), (4, 5), 
    (5, 6), (5, 7), (6, 7), (7, 8), (8, 9), 
    (7, 9), (9, 10)
]
G.add_edges_from(edges)

# Calculate centralities
closeness = nx.closeness_centrality(G)
betweenness = nx.betweenness_centrality(G)
pagerank = nx.pagerank(G)

# Normalize centralities for uniform visualization
norm_closeness = np.array(list(closeness.values()))
norm_closeness = (norm_closeness - norm_closeness.min()) / (norm_closeness.max() - norm_closeness.min())

norm_betweenness = np.array(list(betweenness.values()))
norm_betweenness = (norm_betweenness - norm_betweenness.min()) / (norm_betweenness.max() - norm_betweenness.min())

norm_pagerank = np.array(list(pagerank.values()))
norm_pagerank = (norm_pagerank - norm_pagerank.min()) / (norm_pagerank.max() - norm_pagerank.min())

# Draw the graph with nodes sized by centrality
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
layouts = nx.spring_layout(G, seed=42)

# Closeness centrality
nx.draw(
    G, pos=layouts, ax=axs[0], with_labels=True, 
    node_size=norm_closeness * 2000 + 300, node_color='skyblue'
)
axs[0].set_title("Closeness Centrality")

# Betweenness centrality
nx.draw(
    G, pos=layouts, ax=axs[1], with_labels=True, 
    node_size=norm_betweenness * 2000 + 300, node_color='lightgreen'
)
axs[1].set_title("Betweenness Centrality")

# PageRank centrality
nx.draw(
    G, pos=layouts, ax=axs[2], with_labels=True, 
    node_size=norm_pagerank * 2000 + 300, node_color='salmon'
)
axs[2].set_title("PageRank Centrality")

plt.savefig('centrality_differences.pdf', bbox_inches='tight')
# Calculate Shannon entropy for each graph and normalize it

entropy_list = []
for key, graph in match_dict.items():
    # Total weight of all edges
    total_weight = sum(weight for _, _, weight in graph.edges(data='weight', default=0))
    
    # Calculate probabilities of edge weights
    probabilities = [weight / total_weight for _, _, weight in graph.edges(data='weight', default=0)]
    
    # Raw Shannon entropy
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    # Calculate maximum entropy (log2 of the number of edges)
    num_edges = len(graph.edges)
    max_entropy = np.log2(num_edges) if num_edges > 0 else 1  # Avoid division by zero
    
    # Normalized entropy
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Append result to the list
    entropy_list.append({'Match': key, 'Entropy': normalized_entropy})

# Convert the results into a DataFrame
df_entropy = pd.DataFrame(entropy_list)

# Plot the normalized entropy values as a boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_entropy, x='Entropy', color='cornflowerblue', showfliers=False)
sns.stripplot(data=df_entropy, x='Entropy', color='black', size=6, jitter=True, alpha=0.7)
plt.xlabel('Entropy (0-1)')
plt.savefig('entropy boxplot.pdf', bbox_inches='tight')
