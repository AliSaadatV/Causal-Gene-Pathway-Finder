import pickle
import gzip
import os
import random
import torch
import numpy as np
import gc
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt


def get_data_for_gene(data, gene_symbol):
    # Filter data for the given gene symbol
    gene_data = [item for item in data if item['SYMBOL'] == gene_symbol]

    # Initialize lists for vectors and labels
    vectors = []
    labels = []

    for item in gene_data:
        # Append the vector
        vectors.append(np.tile(item['embed'], (item['n_case']+item['n_control'], 1)))

        # Create labels based on n_case and n_control
        labels += [1] * item['n_case'] + [0] * item['n_control']

    return np.vstack(vectors), np.array(labels)


def case_control(all_embeds, genes):

    gene_f1_scores = {}
    for gene in genes:

        # Get data for the gene (implement this function)
        X_gene, y_gene = get_data_for_gene(all_embeds, gene)  # X_gene: vectors, y_gene: labels

        # Split into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(X_gene, y_gene, test_size=0.25,
                                                            shuffle=True, stratify=y_gene, random_state=42)

        # Train a Random Forest model (or any other suitable model)
        pca = PCA(n_components=10)
        lr = LogisticRegression(max_iter=1000, random_state=42, solver='saga', penalty='elasticnet', l1_ratio=0.5)

        # Make a pipeline
        pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('pca', pca), ('lr', lr)])

        # Train the pipeline
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)

        # Store the F1 and accuracy score
        gene_f1_scores[gene] = f1_score(y_test, y_pred)
    
    return gene_f1_scores


def get_random_data_for_case_control(data, gene_symbol):
    # Filter data for the given gene symbol
    gene_data = [item for item in data if item['SYMBOL'] == gene_symbol]

    # Initialize lists for vectors and labels
    vectors = []
    labels = []

    for item in gene_data:
        # Append the vector
        rand_embed = np.random.randn(256)
        vectors.append(np.tile(rand_embed, (item['n_case']+item['n_control'], 1)))

        # Create labels based on n_case and n_control
        labels += [1] * item['n_case'] + [0] * item['n_control']

    labels = np.array(labels)
    np.random.shuffle(labels)

    return np.vstack(vectors), labels


def case_control_rand(all_embeds, genes):
    # Random Embbeddings
    N_repeat = 1000

    # Dictionary to store F1 scores for each gene
    gene_f1_scores_rand = {}

    for gene in genes:

        temp_f1_scores = []

        for i in range(N_repeat):
            # Get data for the gene (implement this function)
            X_gene, y_gene = get_random_data_for_case_control(all_embeds, gene)

            # Split into training and testing set
            X_train, X_test, y_train, y_test = train_test_split(X_gene, y_gene, test_size=0.25,
                                                                shuffle=True, stratify=y_gene, random_state=42)

            # Train a Random Forest model (or any other suitable model)
            pca = PCA(n_components=10)
            lr = LogisticRegression(max_iter=1000, random_state=42, solver='saga', penalty='elasticnet', l1_ratio=0.5)

            # Make a pipeline
            pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('pca', pca), ('lr', lr)])

            # Train the pipeline
            pipeline.fit(X_train, y_train)

            # Make predictions
            y_pred = pipeline.predict(X_test)

            temp_f1_scores.append(f1_score(y_test, y_pred))

        # Store the F1 and accuracy score
        gene_f1_scores_rand[gene] = temp_f1_scores

    return gene_f1_scores_rand


def calc_pval(observed_dict, rand_dict, n_perm, genes, epsilon=0.001):
    genes_pval = {}

    for gene in genes:
        genes_pval[gene] = (np.sum(np.array(rand_dict[gene]) >= observed_dict[gene]) + epsilon) / (n_perm + epsilon)

    return genes_pval


def plot_rank(scores_dict, xlabel, ylabel, fig_name):
    jitter_y = 0.0001
    jitter_x = 0

    # Convert the list to a DataFrame
    df_temp = pd.DataFrame(list(scores_dict.items()), columns=['SYMBOL', 'Value'])
    df_temp['Value'] = df_temp['Value'] + np.random.uniform(-jitter_y, jitter_y, len(df_temp))
    # Rank the values
    df_temp['Rank'] = (df_temp['Value']).rank(ascending=False)


    # Create the density plot
    plt.figure(figsize=(10, 6))

    # Initialize lists to store the legend handles manually
    outlier_handle = None
    non_outlier_handle = None

# Plotting without distinction in the loop
    for _, row in df_temp.iterrows():
        if row['Rank'] == 1:
            # Plot outlier without label
            outlier_handle = plt.scatter(row['Rank'], row['Value'], color='red')
        else:
            # Plot non-outlier without label
            non_outlier_handle = plt.scatter(row['Rank'], row['Value'], color='blue')

    # Manually create and order legend
    plt.legend([outlier_handle, non_outlier_handle], ['Outlier', 'Non-Outlier'], frameon=True, framealpha=1, shadow=True, borderpad=1)

    plt.xlabel(f'{xlabel}', fontsize=14)
    plt.ylabel(f'{ylabel}', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Add text annotation for top dots
    
    top_dots = df_temp[df_temp["Value"] == df_temp['Value'].max()]
    for index, row in top_dots.iterrows():
        plt.text(row['Rank'], row['Value'], row["SYMBOL"], ha='left', va='bottom')

    plt.savefig(f'{fig_name}.png', dpi=300)


def plot_rand_dist(observed_dict, rand_dict, genes_pval, score_name, plot_name, adjust, gene_name="IFIH1"):
    data = np.array(rand_dict[gene_name])
    data += np.random.uniform(0, 0.01, len(data))

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=10, alpha=0.6, color='blue', stat='proportion')

    plt.axvline(observed_dict[gene_name], color='darkred', linestyle='dashed', linewidth=2)
    plt.text(observed_dict[gene_name]*adjust, plt.gca().get_ylim()[1]*0.8,
            f"observed {score_name} score={observed_dict[gene_name]:.2f}\np-value={genes_pval[gene_name]:.3f}", ha='left', va='bottom')

    # Customizations for publication quality
    plt.title(f'Expected distribution of {score_name} scores for {gene_name}')
    plt.xlabel(f'{score_name} score')
    plt.ylabel('Probability')

    # Optionally, save the plot to a file
    plt.savefig(f'{plot_name}.png', dpi=300, bbox_inches='tight')


def add_alt_ref(all_embeds, total_cases=120):
    for i in range(len(all_embeds)):
        if all_embeds[i]['n_case'] <= total_cases*0.05:
            all_embeds[i]['status'] = "alt_allele"
        else:
            all_embeds[i]['status'] = "ref_allele"
    
    return all_embeds


def case_only(all_embeds, genes, total_cases = 120):

    len_embed = len(all_embeds[0]["embed"])
    embed_dist_res = {}

    for gene in genes:
        sum_dist = 0
        n_case_gene = 0

        for item in all_embeds:
            if item["SYMBOL"] == gene and item['status']=='ref_allele':
                ref_embed = np.array(item["embed"]).reshape((1, len_embed))

        for item in all_embeds:
            if item["SYMBOL"] == gene and item['status']=='alt_allele':
                alt_embed = np.array(item["embed"]).reshape((1, len_embed))

                sum_dist += item['n_case'] * np.linalg.norm(alt_embed - ref_embed)
                n_case_gene += item['n_case']

        embed_dist_res[gene] = sum_dist / total_cases / n_case_gene**0.5

    return embed_dist_res


def case_only_rand(all_embeds, genes, total_cases=120):
    len_embed = len(all_embeds[0]["embed"])
    embed_dist_res_rand = {}
    N_repeat = 1000

    for gene in genes:
        temp_dists = []
        for i in range(N_repeat):
            n_case_gene = 0
            sum_dist = 0
            ref_embed = np.random.rand(1, len_embed)

            for item in all_embeds:
                if item["SYMBOL"] == gene and item['status']=='alt_allele':
                    alt_embed = np.random.rand(1, len_embed)
                    sum_dist += item['n_case'] * np.linalg.norm(alt_embed - ref_embed)
                    n_case_gene += item['n_case']

            temp_dists.append(sum_dist /total_cases /n_case_gene**0.5)

        embed_dist_res_rand[gene] = temp_dists
    
    return embed_dist_res_rand


def main():
    with gzip.open("all_embeds_450k.pkl.gz", 'rb') as file:
        all_embeds = pickle.load(file)

    genes = [item["SYMBOL"] for item in all_embeds]
    genes = list(set(genes))

    #case-vs-control
    gene_f1_scores = case_control(all_embeds, genes)
    plot_rank(gene_f1_scores, "Gene rank", "F1 scores", "case_control_plot")
    gene_f1_scores_rand = case_control_rand(all_embeds, genes)
    case_control_pvals = calc_pval(gene_f1_scores, gene_f1_scores_rand, n_perm=1000, genes=genes)
    plot_rand_dist(gene_f1_scores, gene_f1_scores_rand, case_control_pvals,
                    "F1", "F1_random", gene_name="IFIH1", adjust=0.5)
    
    #case-only
    all_embeds = add_alt_ref(all_embeds)
    gene_distance_scores = case_only(all_embeds, genes)
    plot_rank(gene_distance_scores, "Gene rank", "Distance scores", "case_only_plot")
    gene_distance_scores_rand = case_only_rand(all_embeds, genes)
    case_only_pvals = calc_pval(gene_distance_scores, gene_distance_scores_rand, n_perm=1000, genes=genes)
    plot_rand_dist(gene_distance_scores, gene_distance_scores_rand, case_only_pvals,
                    "Distance", "Dist_random", gene_name="IFIH1", adjust=0.921)

if __name__ == "__main__":
    main()