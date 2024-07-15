import os
import random
import torch
import numpy as np
import gc
import pandas as pd
from embed_generator import inference_single
import pickle
import gzip

def clean_cache():
  torch.cuda.empty_cache()
  gc.collect()

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calc_embeds(df):
   # save results in a list of dictionaries
    all_embeds = []

    # loop and get embeding
    for index, row in df.iterrows():
        embed = inference_single("hyenadna-medium-450k-seqlen", row['sequence'])

        # referenc embeddings
        #if  np.isnan(row["pos"]):
        if row['status'] == 'ref':
            all_pos = df[df["SYMBOL"]==row["SYMBOL"]].dropna(subset=["var_pos"])["var_pos"].values.astype(int)
            if (embed.shape[1] < np.array(all_pos)).any():
                print(f"row {index} skipped! SYMBOL: {row['SYMBOL']}, var_pos: {row['var_pos']}, ID: {row['ID']}")
                continue
            all_embeds.append({"id":row['ID'],
                            "var_pos":all_pos,
                        "SYMBOL":row['SYMBOL'],
                        "n_case":row['n_cases'],
                        "n_control":row['n_controls'],
                        "embed": torch.mean(embed[0, all_pos, :], dim=0).squeeze().tolist()})

        # mutant embeddings
        else:
            all_pos = int(row["var_pos"])
            if (embed.shape[1] < np.array(all_pos)).any():
                continue
            all_embeds.append({"id":row['ID'],
                            "var_pos":all_pos,
                        "SYMBOL":row['SYMBOL'],
                        "n_case":row['n_cases'],
                        "n_control":row['n_controls'],
                        "embed": embed[0, all_pos, :].tolist()})


        clean_cache()
        if index % 50 == 0:
            print(f"index {index} finished. Saved embedding shape = {len(all_embeds[-1]['embed'])}")
    
    return all_embeds


def main():
    seed_everything()

    df = pd.read_csv("df_samples_withseq.tsv", sep="\t")
    df = df[df['seq_len'] <= 450000]
    id_values = [f'id{i}' for i in range(len(df))]
    df['ID'] = id_values

    all_embeds = calc_embeds(df)

    with gzip.open('all_embeds_450k.pkl.gz', 'wb') as file:
        pickle.dump(all_embeds, file)

if __name__ == "__main__":
   main()