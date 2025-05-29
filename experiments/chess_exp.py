#%%
import sys
import os
# Add the parent directory to the system path to import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.grammaticality_exp import generate_pca_plots_from_datasets

import torch
import random
import sys
import os
import torch
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import numpy as np
import os
import torch
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from tqdm.auto import tqdm

from src.utils.pca_utils import (
    perform_dimensionality_reduction,
    create_plot_dataframe
)
from sonar.models.sonar_text import (
    load_sonar_tokenizer,
)
from src.utils.chess_utils import (
    make_move_illegal,
    make_sequence_illegal,
    generate_legal_chess_sequences,
    make_one_san_move_illegal_by_random_destination
)

# SONAR and fairseq2 imports
from sonar.models.sonar_text import load_sonar_tokenizer
# Plotting
import plotly.express as px
from src.custom_sonar_pipeline import CustomTextToEmbeddingPipeline
import numpy as np
#%%
global DEVICE, RANDOM_STATE, MODEL_NAME, OUTPUT_DIR
# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
MODEL_NAME = "text_sonar_basic_encoder"

OUTPUT_DIR = "../data/"

# %%

with open("../data/game_sequences.txt", "r") as f:
    game_sequences = [line for line in f.read().split("\n") if line.strip()]

illegal_game_sequences = make_sequence_illegal(game_sequences)

datasets = [game_sequences, illegal_game_sequences]
labels = ["correct_chess", "incorrect_chess"]

reduced_embeddings, df, fig_interactive, eigenvectors, grammaticality_direction = generate_pca_plots_from_datasets(datasets=datasets, labels=labels, n_components=3, reduction_method="PCA", return_eigenvectors=True, enable_correctness_direction_analysis=True)
# %%

random_legal_game_sequences = generate_legal_chess_sequences(num_sequences=1000, max_moves_per_sequence=20, use_san=True)
illegal_game_sequences = []
num_corruptions_per_sequence = 10

for seq_str in random_legal_game_sequences:
    original_moves = seq_str.split(" ")
    
    if not original_moves: # Handle case of an empty sequence string
        # illegal_game_sequences.append("") # Or skip
        continue

    num_total_moves_in_seq = len(original_moves)
    # This will be the list of moves we modify
    working_moves_list = list(original_moves) 

    # Identify suitable (non-special) indices and special indices
    non_special_indices = []
    special_indices = []
    for i, move_san in enumerate(working_moves_list):
        is_castling = (move_san == "O-O" or move_san == "O-O-O")
        is_promotion = ('=' in move_san)
        if not is_castling and not is_promotion:
            non_special_indices.append(i)
        else:
            special_indices.append(i)
    
    # Shuffle them to pick randomly if we have more candidates than needed
    random.shuffle(non_special_indices)
    random.shuffle(special_indices)

    # Determine the actual number of corruptions we can make (up to num_corruptions_per_sequence)
    # and collect the distinct indices to corrupt
    indices_actually_corrupted_this_sequence = set()

    # Corrupt non-special moves first
    for idx in non_special_indices:
        if len(indices_actually_corrupted_this_sequence) < num_corruptions_per_sequence:
            working_moves_list[idx] = make_one_san_move_illegal_by_random_destination(working_moves_list[idx])
            indices_actually_corrupted_this_sequence.add(idx)
        else:
            break # Reached desired number of corruptions
            
    # If more corruptions are needed and we haven't reached the target, corrupt special moves
    if len(indices_actually_corrupted_this_sequence) < num_corruptions_per_sequence:
        for idx in special_indices:
            if idx not in indices_actually_corrupted_this_sequence: # Ensure it wasn't somehow already picked (shouldn't happen here)
                if len(indices_actually_corrupted_this_sequence) < num_corruptions_per_sequence:
                    working_moves_list[idx] = make_one_san_move_illegal_by_random_destination(working_moves_list[idx])
                    indices_actually_corrupted_this_sequence.add(idx)
                else:
                    break # Reached desired number of corruptions
    
    # The working_moves_list now contains the accumulated corruptions
    illegal_seq_str = " ".join(working_moves_list)
    illegal_game_sequences.append(illegal_seq_str)

# --- Display some results ---
print(f"Target corruptions per sequence: {num_corruptions_per_sequence}")
print(f"Processed {len(random_legal_game_sequences)} legal sequences.")
print(f"Generated {len(illegal_game_sequences)} illegal sequences.")

for i in range(min(5, len(random_legal_game_sequences))):
    print("-" * 20)
    print(f"Original Legal Seq {i+1}: {random_legal_game_sequences[i]}")
    if i < len(illegal_game_sequences):
        # To see which moves changed, you could compare them one by one
        original_split = random_legal_game_sequences[i].split()
        illegal_split = illegal_game_sequences[i].split()
        changed_indices = [k for k in range(len(original_split)) if original_split[k] != illegal_split[k]]
        print(f"Generated Illegal Seq {i+1}: {illegal_game_sequences[i]} (Corrupted at original indices: {changed_indices})")

# %%

datasets = [random_legal_game_sequences, illegal_game_sequences]
labels = ["correct_random_chess", "incorrect_random_chess"]

reduced_embeddings, df, fig_interactive, eigenvectors, grammaticality_direction = generate_pca_plots_from_datasets(datasets=datasets, labels=labels, n_components=3, reduction_method="PCA", return_eigenvectors=True, enable_correctness_direction_analysis=True)
# %%
