
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx



df = pd.read_csv('/Users/MathildeStouby/Desktop/P5 GitHub/5-semester/Fodbold/match_graph_data.csv')

to_drop = df.loc[df['end_position'].isin(['Off T', 'Saved', 'Blocked', 'Goal', 'Wayward','Post', 'Saved to Post', 'Saved Off Target'])].index

df.drop(to_drop, inplace=True)

match_dict = {}
for match_id, group in df.groupby('match_id'):
    # Create graph
    G = nx.DiGraph()
    
    # Add edges with weights
    for _, row in group.iterrows():
        G.add_edge(row['start_position'], row['end_position'], weight=row['pass_frequency'])
    
    # Store the graph in the dictionary with match_id as the key
    match_dict[match_id] = G

# implement the measure

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
df_pagerank = pd.DataFrame(columns=position)


for key, value in match_dict.items():
    pagerank = nx.pagerank(value, weight='weight')
    df_pagerank = pd.concat([df_pagerank, pd.DataFrame(pagerank, index=[key])], ignore_index=True)

na_count = df_pagerank.isna().sum()[:11]
df_subset = df_pagerank[na_count.index]

# make a box plot for each column

position = ['Goalkeeper', 'Left Attacking Midfield', 'Left Defensive Midfield'] 
default_color = "lightgrey"
highlight_color = "cornflowerblue"
custom_palette = {column: (highlight_color if column in position else default_color) for column in df_subset.columns}

oder = ['Goalkeeper', 'Center Back', 'Right Center Back', 'Left Center Back', 'Right Wing Back', 'Left Wing Back', 'Right Defensive Midfield', 'Left Defensive Midfield', 'Right Attacking Midfield', 'Left Attacking Midfield', 'Center Forward']

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_subset, orient='h', palette=custom_palette, showfliers=False, order=oder)
sns.stripplot(data=df_subset, alpha=0.7, color='black', orient='h', size=4, order=oder)
plt.savefig('boxplot.pdf', bbox_inches='tight')

df_closeness = pd.DataFrame(columns=position)
for key, value in match_dict.items():
    closeness = nx.closeness_centrality(value)
    df_closeness = pd.concat([df_closeness, pd.DataFrame(closeness, index=[key])], ignore_index=True)

na_count = df_closeness.isna().sum().sort_values(ascending=True)[:11]
df_close_sub  = df_closeness[na_count.index]


position = ['Goalkeeper', 'Right Wing Back', 'Right Attacking Midfield'] 
default_color = "lightgrey"
highlight_color = "cornflowerblue"
custom_palette = {column: (highlight_color if column in position else default_color) for column in df_subset.columns}


plt.figure(figsize=(10, 6))
sns.boxplot(data=df_close_sub, orient='h', showfliers=False, palette=custom_palette, order=oder)
sns.stripplot(data=df_close_sub, alpha=0.7, color='black', orient='h', size=4, order=oder)
plt.savefig('boxplot.pdf', bbox_inches='tight')


df_betweness = pd.DataFrame(columns=position)
for key, value in match_dict.items():
    closeness = nx.betweenness_centrality(value)
    df_betweness = pd.concat([df_betweness, pd.DataFrame(closeness, index=[key])], ignore_index=True)

na_count = df_betweness.isna().sum().sort_values(ascending=True)[:11]
df_close_sub  = df_betweness[na_count.index]

position = ['Right Center Back', 'Left Defensive Midfield', 'Goalkeeper'] 
default_color = "lightgrey"
highlight_color = "cornflowerblue"
custom_palette = {column: (highlight_color if column in position else default_color) for column in df_subset.columns}


plt.figure(figsize=(10, 6))
sns.boxplot(data=df_close_sub, orient='h', showfliers=False, palette=custom_palette, order=oder)
sns.stripplot(data=df_close_sub, alpha=0.7, color='black', orient='h', size=4, order=oder)
plt.savefig('boxplot.pdf', bbox_inches='tight')


mathc