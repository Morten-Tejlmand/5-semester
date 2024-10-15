import networkx as nx
from itertools import combinations
from networkx.algorithms.isomorphism import DiGraphMatcher

def frequent_singletons(min_sup, edge_matrix):
    items_counted = {}
    
    for edge_list in edge_matrix:
        for edge in edge_list:
            # Use only source and target nodes for counting
            edge_key = (edge[0], edge[1])
            items_counted[edge_key] = items_counted.get(edge_key, 0) + 1
    
    # Filter edges that meet the min_sup
    F = [key for key, value in items_counted.items() if value >= min_sup]
    
    F_graphs = []
    for edge in F:
        g = nx.DiGraph()
        g.add_edge(edge[0], edge[1])  # Add edge without sequence for the subgraph
        F_graphs.append(g)
    
    return F_graphs


# Generate candidates of size k subgraphs from frequent subgraphs
def generate_candidates(F, k):
    candidates = set()
    
    # Iterate over all pairs of frequent subgraphs (F)
    for g1, g2 in combinations(F, 2):

        edges_g1 = sorted(g1.edges(data=True), key=lambda e: e[2].get('sequence', 0))
        edges_g2 = sorted(g2.edges(data=True), key=lambda e: e[2].get('sequence', 0))
        
        if edges_g1[2].get('sequence') > edges_g2[2].get('sequence'):
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
                print(f"Edge {u} -> {v}, Attributes: {attr}")
        
        k += 1  # Move to the next size of subgraphs

    return frequent_total

