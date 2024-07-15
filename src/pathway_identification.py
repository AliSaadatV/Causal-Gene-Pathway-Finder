import pandas as pd
import numpy as np
import pickle
import gzip
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric
from torch_geometric.nn import GCNConv, global_sort_pool
import torch.nn.init as init
import torch.nn as nn
from sklearn.metrics import f1_score
from torch_geometric.explain import Explainer, GNNExplainer
import gseapy as gp
import random


def read_and_filter_df(path):
    df = pd.read_csv(path, sep="\t")
    df = df[df['seq_len'] <= 450000]
    id_values = [f'id{i}' for i in range(len(df))]
    df['ID'] = id_values
    return df

def read_and_filter_df_ppi(df, genes, path):
    df_ppi = pd.read_csv(path, sep="\t")
    df_ppi = df_ppi.rename(columns={'#node1': 'node1'})
    df_ppi = df_ppi.loc[:, ["node1", "node2", "combined_score"]]
    df_ppi = df_ppi[df_ppi["combined_score"] >= 0.6]
    df_ppi = df_ppi[df_ppi.apply(lambda row: row['node1'] in df["SYMBOL"].values and row['node2'] in df["SYMBOL"].values, axis=1)]
    df_ppi = df_ppi[df_ppi.apply(lambda row: row['node1'] in genes and row['node2'] in genes, axis=1)]

    G = nx.from_pandas_edgelist(df_ppi, 'node1', 'node2', ['combined_score']) # Step 1: Create a graph from the DataFrame
    largest_cc = max(nx.connected_components(G), key=len) # Step 2: Find the largest connected component
    subgraph = G.subgraph(largest_cc) # Step 3: Extract the subgraph that corresponds to the largest connected component
    df_ppi = nx.to_pandas_edgelist(subgraph) # Step 4: Convert the subgraph back to DataFrame
    df_ppi.columns = ['node1', 'node2', 'combined_score']

    return df_ppi


def get_node_and_edges(df_ppi):
    node1 = df_ppi['node1']
    node2 = df_ppi['node2']
    scores = df_ppi['combined_score']

    # Create Node Index
    nodes = pd.unique(df_ppi[['node1', 'node2']].values.ravel('K'))
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_to_node = {idx: node for idx, node in enumerate(nodes)}

    # Create Edge Index and Edge Features
    edge_sources = torch.tensor([node_to_idx[node] for node in node1], dtype=torch.long)
    edge_targets = torch.tensor([node_to_idx[node] for node in node2], dtype=torch.long)
    edge_index = torch.stack([edge_sources, edge_targets], dim=0)

    edge_features = torch.tensor(scores.values, dtype=torch.float).view(-1, 1)
    
    return edge_index, edge_features, node_to_idx, idx_to_node, nodes


def get_unique_ids(df, colname, sep=","):

  df[colname] = df[colname].fillna('').astype(str)

  # Step 1 & 2: Split the "samples" column and flatten it into a single list
  all_ids = [id.strip() for sublist in df[colname].str.split(',') for id in sublist]

  # Step 3: Convert the list to a set to get unique IDs
  all_ids = list(set(all_ids))

  return [id for id in all_ids if id]


def get_node_feature(df_cp, sample_id, colname, nodes, all_embeds):
  df_cp[colname] = df_cp[colname].astype(str)
  filtered_df = df_cp[df_cp[colname].str.contains(r'\b'+ sample_id + r'\b', na=False, regex=True)]

  x = np.zeros((len(nodes), len(all_embeds[0]["embed"])))

  for i in range(len(nodes)):
    gene = nodes[i]
    for item in all_embeds:
      if (item["SYMBOL"]==gene) and (item["id"]==filtered_df[filtered_df["SYMBOL"]==gene]["ID"].values[0]):
        x[i, :] = np.array(item["embed"])

  return torch.Tensor(x)


class GCNSortPool(torch.nn.Module):
    def __init__(self, num_node_features=256, dim_h1=16, dim_h2=16, output_dim=1, k=40):
        super(GCNSortPool, self).__init__()
        self.conv1 = GCNConv(num_node_features, dim_h1)
        self.conv2 = GCNConv(dim_h1, dim_h2)
        self.k = k  # The number of nodes to keep after sorting.
        self.fc = torch.nn.Linear(dim_h2*k, output_dim)

    def forward(self, x, edge_index, batch, edge_weight):

        # Apply GCN layers
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))

        # Use global_sort_pool to aggregate node features
        x = global_sort_pool(x, batch, self.k)

        # A final fully connected layer for classification
        x = self.fc(x)

        return x


def pos_weight(train_loader, device):
    num_pos_samples = sum(data.y for data in train_loader.dataset)
    num_neg_samples = len(train_loader.dataset) - num_pos_samples
    pos_weight = num_neg_samples / num_pos_samples

    # Ensure pos_weight is a tensor of the correct type and device
    return torch.tensor([pos_weight], dtype=torch.float, device=device)


def train_model(model, train_loader, optimizer, criterion, device, threshold=0.5, n_epochs=1000):
   for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    y_true = []
    y_pred = []

    for data in train_loader:  # Assuming each batch is a Data object from PyTorch Geometric
        data.to(device)
        optimizer.zero_grad()
        outputs = model(data.x, data.edge_index, data.batch, data.edge_attr.squeeze())
        loss = criterion(outputs.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        y_true.extend(data.y.tolist())
        y_pred.extend(outputs.detach().cpu().view(-1).numpy())

    if epoch % 50 ==0:
        # Convert logits to binary predictions
        y_pred_binary = [1 if prob > threshold else 0 for prob in torch.sigmoid(torch.tensor(y_pred))]

        # Calculate metrics
        accuracy = np.mean(np.array(y_true) == np.array(y_pred_binary))
        f1 = f1_score(y_true, y_pred_binary)
        print(f'Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}, Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}')


def get_explanations(model, data_list):
   
    explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=250),
            explanation_type='model',
            node_mask_type='object',
            edge_mask_type='object',
            model_config=dict(
                mode='binary_classification',
                task_level='graph',
                return_type='raw',
            ),
        )
    
    explanation = explainer(x=data_list[0].x, edge_index=data_list[0].edge_index, target=torch.Tensor(data_list[0].y),
                             batch=data_list[0].batch, edge_weight=data_list[0].edge_attr.squeeze())
   
    sum_node_masks = torch.zeros(explanation["node_mask"].reshape(-1, ).shape)
    sum_edge_masks = torch.zeros(explanation["edge_mask"].shape)

    for i in range(len(data_list)):
        temp_explanation = explainer(x=data_list[i].x, edge_index=data_list[i].edge_index, target=torch.Tensor(data_list[i].y),
                                      batch=data_list[i].batch, edge_weight=data_list[i].edge_attr.squeeze())
        sum_node_masks += temp_explanation["node_mask"].reshape(-1, )
        sum_edge_masks += temp_explanation["edge_mask"]

    mean_node_masks  = sum_node_masks/ (len(data_list))
    mean_edge_masks = sum_edge_masks / (len(data_list))

    return mean_edge_masks, mean_node_masks


def get_GA_input(mean_node_masks, mean_edge_masks, data_list, nodes):
    node_scores = {}
    for i in range(len(nodes)):
        node_scores[i] = mean_node_masks[i].item()

    edge_weights = mean_edge_masks.numpy()
    edge_index = data_list[1].edge_index

    G = nx.Graph()
    for i, (start, end) in enumerate(edge_index.t().tolist()):
        G.add_edge(start, end, weight=edge_weights[i].item())

    for node in G.nodes:
        G.nodes[node]['score'] = node_scores[node]

    nodes_with_score_above_zero = [node for node, attr in G.nodes(data=True) if attr['score'] > 0]
    G = G.subgraph(nodes_with_score_above_zero).copy()  # Create a subgraph and copy it to make it independent
    edges_to_remove = [(u, v) for u, v in G.edges if G.nodes[u]['score'] <0.001 or G.nodes[v]['score'] <0.001]
    G.remove_edges_from(edges_to_remove)

    return G


class Individual:
    def __init__(self, nodes):
        self.nodes = nodes
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        # Fitness is the sum of scores of included nodes
        return sum(G.nodes[node]['score'] for node in self.nodes) / (len(self.nodes))

    def is_connected(self):
        # Check if the subgraph defined by self.nodes is connected
        subgraph = G.subgraph(self.nodes)
        return nx.is_connected(subgraph)

def generate_initial_individual():
    node = random.choice(list(G.nodes))
    nodes = {node}
    for _ in range(random.randint(1, len(G.nodes) // 2)):  # Random subgraph size
        neighbors = list(nx.neighbors(G, node))
        if neighbors:
            node = random.choice(neighbors)
            nodes.add(node)
    return Individual(nodes)

def crossover(parent1, parent2):
    common_nodes = parent1.nodes.intersection(parent2.nodes)
    if common_nodes:
        crossover_node = random.choice(list(common_nodes))
        offspring_nodes = {crossover_node}
        for parent in [parent1, parent2]:
            for node in nx.dfs_preorder_nodes(G.subgraph(parent.nodes), source=crossover_node):
                offspring_nodes.add(node)
        return Individual(offspring_nodes)
    else:
        return random.choice([parent1, parent2])

def mutate(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        mutation_node = random.choice(list(individual.nodes))
        neighbors = list(nx.neighbors(G, mutation_node))
        if neighbors:
            new_node = random.choice(neighbors)
            individual.nodes.add(new_node)
            if not individual.is_connected():
                individual.nodes.remove(new_node)
    return individual

def genetic_algorithm(G, population_size=100, generations=5, mutation_rate=0.1):
    population = [generate_initial_individual() for _ in range(population_size)]

    for generation in range(generations):
        # Evaluate fitness
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        # Selection - Top 50% survive
        weights = [ind.fitness for ind in population]
        selected = random.choices(population, weights=weights, k=len(population)//2)
        #selected = population[:len(population)//2]

        # Crossover and mutation to generate new individuals
        offspring = []
        while len(offspring) < population_size // 2:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            if child.is_connected():
                offspring.append(child)

        population = selected + offspring

    # Return the best individual
    best_individual = max(population, key=lambda ind: ind.fitness)
    return best_individual, population


def main():
    # Read files
    with gzip.open("all_embeds_450k.pkl.gz", 'rb') as file:
        all_embeds = pickle.load(file)

    genes = [item["SYMBOL"] for item in all_embeds]
    genes = list(set(genes))

    df = read_and_filter_df("df_samples_noseq.tsv")
    df_ppi = read_and_filter_df_ppi(df, genes, "pri_interactions_short.tsv")
    
    # get edge and nodes from PPI
    edge_index, edge_features, node_to_idx, idx_to_node, nodes = get_node_and_edges(df_ppi)

    # keep genes that are inside ppi
    df = df[df.apply(lambda row: row['SYMBOL'] in nodes , axis=1)] 

    # get unique individual IDs
    cases_ids = get_unique_ids(df, "samples_case", ",")
    controls_ids = get_unique_ids(df, "samples_cont", ",")

    # create datalist
    data_list = []

    for case_id in cases_ids:
        x = get_node_feature(df.copy(), case_id, "samples_case", nodes, all_embeds)
        data = Data(x=x, y=1, edge_index=edge_index, edge_attr=edge_features)
        data_list.append(data)

    for cont_id in controls_ids:
        x = get_node_feature(df.copy(), cont_id, "samples_cont", nodes, all_embeds)
        data = Data(x=x, y=0, edge_index=edge_index, edge_attr=edge_features)
        data_list.append(data)


    # create data loader
    train_loader = DataLoader(data_list, batch_size=32, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pos weight calculation
    pos_weight = pos_weight(train_loader, device)

    # initialize model
    model = GCNSortPool(num_node_features=data_list[0].x.shape[1], dim_h1=16, dim_h2=16, k=40, output_dim=1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # train the model
    train_model(model, train_loader, optimizer, criterion, device)

    # get explanations
    mean_edge_masks, mean_node_masks = get_explanations(model, data_list)

    # get genetic algorithm (GA) input
    input_graph_GA = get_GA_input(mean_node_masks, mean_edge_masks, data_list, nodes)

    # run GA
    best_subgraph, population = genetic_algorithm(input_graph_GA, population_size=100, generations=10, mutation_rate=0.5)

    selected_subnetwork = [idx_to_node[i] for i in best_subgraph.nodes]

    # pathway enrichment results
    enr = gp.enrichr(gene_list=list(selected_subnetwork),
                 gene_sets=['Reactome_2022'],
                 organism='human',
                )

    enr.results[enr.results['Adjusted P-value'] <= 0.05]



if __name__ == "__main__":
    main()