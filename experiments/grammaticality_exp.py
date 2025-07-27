# %%
import os
import random
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import umap
import phate
from tqdm.auto import tqdm

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split

from sonar.models.sonar_text import load_sonar_tokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.custom_sonar_pipeline import CustomTextToEmbeddingPipeline
from src.utils.generate_random_sequences import generate_random_sequences
from src.utils.pca_utils import (
    create_plot_dataframe,
    generate_pca_plots_from_datasets,
    perform_dimensionality_reduction,
)
#%%
global DEVICE, RANDOM_STATE, MODEL_NAME, OUTPUT_DIR, FIGURES_DIR
# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
MODEL_NAME = "text_sonar_basic_encoder"

OUTPUT_DIR = "../data/"
FIGURES_DIR = "../figures/"

#%%

reduced_embeddings, df, fig_interactive, eigenvectors, grammaticality_direction = generate_pca_plots_from_datasets(datasets= None, datasets_paths=["../data/datasets/grammar_english.txt", "../data/datasets/grammar_german.txt", "../data/datasets/grammar_catalan.txt", "../data/datasets/agrammar_english.txt", "../data/datasets/agrammar_german.txt", "../data/datasets/agrammar_catalan.txt"], labels=["grammatical_language", "grammatical_language", "grammatical_language", "agrammatical_language", "agrammatical_language", "agrammatical_language"], n_components=2, reduction_method="PCA", return_eigenvectors=True, enable_correctness_direction_analysis=True)

# %%
import random

def create_agrammatical_sentences(input_file, output_file):
    # Read grammatical sentences from the file
    with open(input_file, "r") as f:
        grammatical_sentences = [line.strip() for line in f if line.strip()]
    
    print(f"Read {len(grammatical_sentences)} grammatical sentences")
    
    # Create agrammatical sentences by shuffling words
    agrammatical_sentences = []
    for sentence in grammatical_sentences:
        # Remove the period at the end
        if sentence.endswith('.'):
            base_sentence = sentence[:-1]
        else:
            base_sentence = sentence
            
        # Split into words and lowercase all words
        words = [word.lower() for word in base_sentence.split()]
        
        # Shuffle the words
        random.shuffle(words)
        
        # Capitalize the first word
        if words:
            words[0] = words[0].capitalize()
        
        # Join words back together and add period
        agrammatical_sentence = ' '.join(words) + '.'
        agrammatical_sentences.append(agrammatical_sentence)
    
    # Write agrammatical sentences to a new file
    with open(output_file, "w") as f:
        f.write('\n'.join(agrammatical_sentences))
    
    print(f"Successfully wrote {len(agrammatical_sentences)} agrammatical sentences to {output_file}")
    
    # Print a few examples for verification
    print("\nSample comparison:")
    for i in range(min(5, len(grammatical_sentences))):
        print(f"Original: {grammatical_sentences[i]}")
        print(f"Shuffled: {agrammatical_sentences[i]}")
        print()

# Set random seed for reproducibility
random.seed(42)

# Process the file
create_agrammatical_sentences("../data/datasets/grammar_english.txt", "../data/datasets/agrammar_english_shuffled.txt")
# %%


reduced_embeddings, df, fig_interactive, eigenvectors, grammaticality_direction = generate_pca_plots_from_datasets(datasets= None, datasets_paths=["../data/datasets/grammar_english.txt", "../data/datasets/agrammar_english_shuffled.txt"], labels=["grammar_english", "agrammar_english"], n_components=2, reduction_method="UMAP", return_eigenvectors=True, enable_correctness_direction_analysis=True)

# %%