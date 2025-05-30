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
    make_one_san_move_illegal_by_random_destination,
    generate_illegal_game_sequences,
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

# %%
max_moves_per_sequence = 10
num_sequences = 1000
num_corruptions_per_sequence = 5
random_legal_game_sequences = generate_legal_chess_sequences(num_sequences=num_sequences, max_moves_per_sequence=max_moves_per_sequence, use_san=True)

illegal_game_sequences = generate_illegal_game_sequences(random_legal_game_sequences, num_corruptions_per_sequence=num_corruptions_per_sequence)

# for i in range(len(random_legal_game_sequences)):
#     random_legal_game_sequences[i] = "This is a SAN chess game playout: " + random_legal_game_sequences[i]
    
# for i in range(len(illegal_game_sequences)):
#     illegal_game_sequences[i] = "This is a SAN chess game playout: " + illegal_game_sequences[i]

# Add a prefix to the sequences to indicate they are SAN chess game playouts
# %%

human_game_sequences = []

for seq in game_sequences:
    moves = seq.split(" ")
    if len(moves) < max_moves_per_sequence:
        continue # Skip sequences that are too short
    
    human_game_sequences.append(" ".join(moves[:max_moves_per_sequence]))
    
human_game_sequences = human_game_sequences[:num_sequences]

illegal_human_game_sequences = generate_illegal_game_sequences(human_game_sequences, num_corruptions_per_sequence=num_corruptions_per_sequence)


# %%

datasets = [random_legal_game_sequences, illegal_game_sequences]
labels = ["random_legal_game_sequences", "illegal_game_sequences"]

reduced_embeddings, df, fig_interactive, eigenvectors, grammaticality_direction = generate_pca_plots_from_datasets(datasets=datasets, labels=labels, n_components=2, reduction_method="PCA", return_eigenvectors=True, enable_correctness_direction_analysis=True)
# %%
