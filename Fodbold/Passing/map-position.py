from statsbombpy import sb
import pandas as pd
import networkx as nx
import networkx as nx
from itertools import combinations
import multiprocessing

from networkx.algorithms.isomorphism import DiGraphMatcher
multiprocessing.set_start_method('fork', force=True)
from helper_functions import(match_ids,
                             create_graphs,
                             create_graphs_dict,
                             plot_graphs)
events = sb.competition_events(
    country="Germany",
    division= "1. Bundesliga",
    season="2023/2024",
    gender="male"
)

df = match_ids(events, "Bayer Leverkusen", season_id=281, competition_id=9)
# filer so match id is 3895348
df = df[df["match_id"] == 3895348]
df_lineup = df[df["tactics"].notna()]
df_lineup['tactics'] = df_lineup["tactics"]
df_lineup["formation"] = df_lineup["tactics"].apply(lambda x: x["formation"])
df_lineup['xg'] = df_lineup.groupby('formation')['shot_statsbomb_xg'].transform(lambda group: group.fillna(method='ffill').fillna(method='ffill'))
# Add number and timestamp
df_lineup["number"] = (df_lineup["formation"] != df_lineup["formation"].shift()).cumsum()


df_lineup['tactics'] = df_lineup["tactics"]
df_lineup["formation"] = df_lineup["tactics"].apply(lambda x: x["formation"])
df_lineup["lineup"] = df_lineup["tactics"].apply(lambda x: x["lineup"])
df_tactics = df_lineup
    



for row in df.iterrows():
    possesion_team = row[1]["possession_team"]
    if possesion_team != "Bayer Leverkusen":
        continue
    
    player_id = row[1]["player_id"]
    
    if pd.isna(player_id):
        continue

    player_id = int(player_id)

    for i in row["tactics"]["lineup"]:

        df_tactics["position_id"] = row["position"]["id"]
        df_tactics["position_name"] = row["position"]["name"]
    if player_id:
        print(player_id)


test = df_tactics["lineup"][0]
for i in test:
    i
    possession_id = i["pos"]["id"]
    
    