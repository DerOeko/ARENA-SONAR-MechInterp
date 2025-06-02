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
import numpy as np
import os
import torch
from src.utils.get_embeddings_similarity import calculate_embedding_list_similarity
from tqdm.auto import tqdm
from sonar.models.sonar_text import (
    load_sonar_tokenizer,
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

#%%

num_samples = 1000

X_Y = []
Z = []
for _ in range(num_samples):
    # Generate a random math expression with 4 digits and basic operations
    digits = [random.randint(1000, 9999) for _ in range(3)]
    X_Y.append(f"{digits[0]} + {digits[1]}")
    Z.append(f"{digits[0] + digits[1]}")
    #Z.append(f"{digits[0] + digits[1] + random.randint(-1000, 1000)}")  # Adding some noise to the result


print(f"\n--- Calculating Embedding Similarity for X_Y vs Z ({num_samples} samples) ---")
try:
    similarity_results = calculate_embedding_list_similarity(
        dataset1_texts=X_Y,
        dataset2_texts=Z,
        model_name=MODEL_NAME,
        device_override=DEVICE,
        random_seed=RANDOM_STATE,
    )

    print("\nSimilarity Analysis Results:")
    print(f"  Overall similarity [Avg(X_Y) vs Avg(Z)]: {similarity_results['similarity_avg_dataset1_vs_avg_dataset2']:.4f}")
    print(f"  Number of embeddings in X_Y: {similarity_results['num_embeddings_dataset1']}")
    print(f"  Number of embeddings in Z: {similarity_results['num_embeddings_dataset2']}")
    print(f"  Uniform padding length used: {similarity_results['uniform_padding_length_used']}")

    if "avg_similarity_of_individual_dataset2_vs_avg_dataset1" in similarity_results:
        print(f"  Average of individual Z sentence similarities to Avg(X_Y): {similarity_results['avg_similarity_of_individual_dataset2_vs_avg_dataset1']:.4f}")

except Exception as e:
    print(f"An error occurred during the similarity calculation: {e}")