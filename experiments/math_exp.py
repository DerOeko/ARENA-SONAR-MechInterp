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
from src.utils.get_embeddings_similarity import calculate_embedding_list_similarity
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
    generate_data_for_probing
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


num_samples = 1000

incorrect_math_sequences = []
correct_math_sequences = []
for _ in range(num_samples):
    # Generate a random math expression with 4 digits and basic operations
    digits = [random.randint(1000, 9999) for _ in range(3)]
    correct_math_sequences.append(f"The result of {digits[0]} + {digits[1]} is {digits[0] + digits[1]}")
    incorrect_math_sequences.append(f"The result of {digits[0]} + {digits[1]} is {digits[2]}")
# %%


datasets = [correct_math_sequences, incorrect_math_sequences]
labels = ["correct_math_sequences", "incorrect_math_sequences"]

reduced_embeddings, df, fig_interactive, eigenvectors, grammaticality_direction = generate_pca_plots_from_datasets(datasets=datasets, labels=labels, n_components=2, reduction_method="PCA", return_eigenvectors=True, enable_correctness_direction_analysis=True)
# %%

num_samples = 1000

X_Y = []
Z = []
for _ in range(num_samples):
    # Generate a random math expression with 4 digits and basic operations
    digits = [random.randint(1000, 9999) for _ in range(3)]
    X_Y.append(f"{digits[0]} + {digits[1]}")

    Z.append(f"{digits[0] + digits[1] + random.randint(-1000, 1000)}")  # Adding some noise to the result


print(f"\n--- Calculating Embedding Similarity for X_Y vs Z ({num_samples} samples) ---")
try:
    similarity_results = calculate_embedding_list_similarity(
        dataset1_texts=X_Y,
        dataset2_texts=Z,
        model_name=MODEL_NAME,        # Using your global MODEL_NAME
        device_override=DEVICE,       # Using your global DEVICE
        random_seed=RANDOM_STATE,     # Using your global RANDOM_STATE
        # You can also set other parameters like:
        # inference_batch_size=128, # (default is often 128)
        # return_individual_dataset2_similarities=True, # (default is True)
        # return_average_embeddings_data=False # (default is False)
    )

    print("\nSimilarity Analysis Results:")
    print(f"  Overall similarity [Avg(X_Y) vs Avg(Z)]: {similarity_results['similarity_avg_dataset1_vs_avg_dataset2']:.4f}")
    print(f"  Number of embeddings in X_Y: {similarity_results['num_embeddings_dataset1']}")
    print(f"  Number of embeddings in Z: {similarity_results['num_embeddings_dataset2']}")
    print(f"  Uniform padding length used: {similarity_results['uniform_padding_length_used']}")

    if "avg_similarity_of_individual_dataset2_vs_avg_dataset1" in similarity_results:
        print(f"  Average of individual Z sentence similarities to Avg(X_Y): {similarity_results['avg_similarity_of_individual_dataset2_vs_avg_dataset1']:.4f}")

    # Individual similarities (if returned):
    # if similarity_results.get("individual_dataset2_similarities_vs_avg_dataset1"):
    #     print("\n  First 3 individual Z similarities to Avg(X_Y):")
    #     for item in similarity_results["individual_dataset2_similarities_vs_avg_dataset1"][:3]:
    #         print(f"    Text: \"{item['text_dataset2']}\", Similarity: {item['similarity_to_avg_datase        t1']:.4f}")

except Exception as e:
    print(f"An error occurred during the similarity calculation: {e}")

# %%
