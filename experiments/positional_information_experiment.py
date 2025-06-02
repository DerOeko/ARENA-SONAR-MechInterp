#%%
import torch
import random
import itertools
import matplotlib.pyplot as plt 
import numpy as np
import os
from tqdm import tqdm
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from fairseq2.data import Collater
from fairseq2.data.data_pipeline import read_sequence
from fairseq2.data.text import TextTokenizer, read_text
from fairseq2.generation import (
    BeamSearchSeq2SeqGenerator,
    Sampler,
    SamplingSeq2SeqGenerator,
    Seq2SeqGenerator,
    SequenceToTextConverter,
    TextTranslator,
)
from fairseq2.typing import CPU, DataType, Device

from sonar.inference_pipelines.utils import add_progress_bar, extract_sequence_batch
from sonar.models.encoder_model import SonarEncoderModel
from sonar.models.sonar_text import (
    load_sonar_text_decoder_model,
    load_sonar_text_encoder_model,
    load_sonar_tokenizer,
)
from sonar.models.sonar_translation import SonarEncoderDecoderModel
from sonar.models.sonar_translation.model import DummyEncoderModel
from sonar.nn.conditional_decoder_model import ConditionalTransformerDecoderModel
# SONAR and fairseq2 imports
from sonar.models.sonar_text import load_sonar_tokenizer
from sonar.models.encoder_model import SonarEncoderModel # For type hinting
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, extract_sequence_batch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask
from fairseq2.data import Collater 
from fairseq2.typing import Device, DataType, CPU
from fairseq2.models.sequence import SequenceModelOutput
from typing import Optional, Tuple
from src.custom_sonar_pipeline import CustomTextToEmbeddingPipeline
# Plotting
import plotly.express as px
import pandas as pd
from phate import PHATE
global VOCAB_INFO, PAD_IDX, EOS_IDX, UNK_IDX, ENG_LANG_TOKEN_IDX, DEVICE, RANDOM_STATE, MODEL_NAME, OUTPUT_DIR

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
MODEL_NAME = "text_sonar_basic_encoder"

WORDS_TO_TEST = ["dog", "cat", "car", "house", "tree", "love", "run", "code", "data", "model"]

OUTPUT_DIR = "./data/"

# Seed for reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(RANDOM_STATE)

# %% --- Tokenizer and Special Token IDs ---
print("--- Initializing Tokenizer and Special IDs ---")
orig_sonar_tokenizer = load_sonar_tokenizer(MODEL_NAME)
tokenizer_encoder = orig_sonar_tokenizer.create_encoder()
tokenizer_decoder = orig_sonar_tokenizer.create_decoder()

VOCAB_INFO = orig_sonar_tokenizer.vocab_info
PAD_IDX = VOCAB_INFO.pad_idx
EOS_IDX = VOCAB_INFO.eos_idx
UNK_IDX = VOCAB_INFO.unk_idx
dummy_tokenized_for_special_tokens = tokenizer_encoder("test")
ENG_LANG_TOKEN_IDX = dummy_tokenized_for_special_tokens[0].item()

print(f"Using Language ID (eng_Latn): {ENG_LANG_TOKEN_IDX} ('{tokenizer_decoder(torch.tensor([ENG_LANG_TOKEN_IDX]))}')")
print(f"Using PAD ID: {PAD_IDX} ('{tokenizer_decoder(torch.tensor([PAD_IDX]))}')")

#%%
text2vec = CustomTextToEmbeddingPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=DEVICE
)
MAX_SEQ_LEN = text2vec._modules['model'].encoder_frontend.pos_encoder.max_seq_len

# testing the pipeline
test_sentence = "The quick brown fox jumps over the lazy dog."

#test_encoded_list = tokenizer_encoder(test_sentence) 

#print(f"Tokenized sentence (tensor): {test_encoded_list}")

# Add batch dimension for predict_from_token_ids as it expects [batch_size, seq_len]
#test_embedding = text2vec.predict_from_token_ids(test_encoded_list) 

# Load Sonar encoder and generate embeddings for tokenized sentences
print("\n--- Generating Embeddings for Sentences ---")

words_to_test = ["house"]
word_token_ids = {word: tokenizer_encoder(word)[1] for word in words_to_test}
print(f"Token IDs for words {words_to_test}: {word_token_ids}")

all_token_sequences = []
all_labels = []
all_positions = []
for word_str in words_to_test:
    for i in range(1, MAX_SEQ_LEN - 1, 4):
        token_ids = torch.full((MAX_SEQ_LEN,), UNK_IDX, dtype=torch.long, device= DEVICE)
        token_ids[0] = 256047
        token_ids[-1] = EOS_IDX
        token_ids[i] = word_token_ids[word_str]
        
        all_token_sequences.append(token_ids)
        all_labels.append(f"{word_str}_pos{i}")
        all_positions.append(i)

# Convert to tensor
all_token_sequences = torch.stack(all_token_sequences).to(DEVICE)
all_positions = torch.tensor(all_positions, dtype=torch.long, device=DEVICE)#%% Get word embeddings
#%%
INFERENCE_BATCH_SIZE = 96
sentence_embeddings_list = [] # Storing sentence embeddings (previously word_embeddings)
last_token_embeddings_list = []    # Storing specific token hidden states (previously token_embeddings)
first_token_embeddings_list = [] # Storing first token hidden states (if needed)
with torch.no_grad():
    # Corrected loop iterator
    for batch_start_idx in tqdm(range(0, len(all_token_sequences), INFERENCE_BATCH_SIZE), desc="Generating Embeddings"):
        batch_end_idx = batch_start_idx + INFERENCE_BATCH_SIZE # Calculate end index for slicing
        
        batch_tokens_ids = all_token_sequences[batch_start_idx:batch_end_idx]
        batch_actual_positions = all_positions[batch_start_idx:batch_end_idx] # These are the 'i' values for target tokens

        if batch_tokens_ids.size(0) == 0: # Handle potential empty last batch if len % BATCH_SIZE == 0
            continue

        # Get sentence embeddings and all last hidden states for the batch
        batch_sentence_embeddings, batch_last_hidden_states, batch_first_hidden_states, _ = text2vec.predict_from_token_ids(batch_tokens_ids)
        
        # Store sentence embeddings (optional, if you still need them)
        sentence_embeddings_list.append(batch_sentence_embeddings.cpu().numpy())
        
        # Extract the hidden state of the target token using its actual position in the sequence
        # batch_indices will be [0, 1, ..., current_batch_size-1]
        batch_indices = torch.arange(batch_tokens_ids.size(0), device=DEVICE)
        
        # This is the key step:
        # batch_last_hidden_states has shape [current_batch_size, MAX_SEQ_LEN, hidden_dim]
        # batch_indices has shape [current_batch_size]
        # batch_actual_positions (your previous batch_word_ids) has shape [current_batch_size]
        # This correctly selects the hidden state vector for the target token at its specific position 'i'
        # for each sequence in the batch.
        specific_token_last_embeddings_batch = batch_last_hidden_states[batch_indices, batch_actual_positions]
        last_token_embeddings_list.append(specific_token_last_embeddings_batch.cpu().numpy())
        
        specific_token_initial_embeddings_batch = batch_first_hidden_states[batch_indices, batch_actual_positions]
        first_token_embeddings_list.append(specific_token_initial_embeddings_batch.cpu().numpy())
# Moved print statement outside the loop
print(f"Generated embeddings for words: {words_to_test}")

# Now, last_token_embeddings_list contains a list of numpy arrays,
# each array being the hidden states of your target token (e.g., "dog")
# from a batch. You'll concatenate them for PCA/PHATE.

# Example of concatenating for PCA:
if last_token_embeddings_list: # Check if the list is not empty
    embeddings_for_pca = np.concatenate(last_token_embeddings_list, axis=0)
    # Now 'embeddings_for_pca' is ready for your PCA and plotting code.
    # Ensure all_labels, words_for_df, positions_for_df align with this concatenated array.
else:
    print("No embeddings were generated.")
#%%
import torch # Needed for isinstance(all_positions, torch.Tensor) in create_dataframe
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import numpy as np
import os

# --- Helper Functions (from your code, slightly adjusted for clarity/consistency) ---
# Ensure RANDOM_STATE is defined globally if used by perform_pca, e.g., RANDOM_STATE = 42
# Ensure OUTPUT_DIR is defined globally if used by the calling code.


#%%
# --- Process and Plot LAST Token Embeddings (Final Hidden States) ---
if last_token_embeddings_list:
    print("\n\n--- Processing LAST Token Embeddings (Final Hidden States) ---")
    embeddings_final_concatenated = prepare_embeddings_for_pca(last_token_embeddings_list)
    
    # For 2D PCA Plot
    reduced_embeddings_2d_final, last_token_eigenvectors = perform_dimensionality_reduction(
        embeddings_final_concatenated, method="pca", n_components=2, random_state=RANDOM_STATE
    )
    words_for_df_final = [label.split('_pos')[0] for label in all_labels] # Recalculate or pass if already available
    df_plot_2d_final = create_plot_dataframe(
        reduced_embeddings_2d_final, all_labels, all_positions, words_for_df_final
    )
    plot_dimensionality_reduction_results(
        df_plot_2d_final, words_to_test, OUTPUT_DIR, "Final_Token_Embeddings", "PCA"
    )

    # For 3D PCA Plot
    reduced_embeddings_3d_final, last_token_eigenvectors3d = perform_dimensionality_reduction(
        embeddings_final_concatenated, method="pca", n_components=3, random_state=RANDOM_STATE
    )
    df_plot_3d_final = create_plot_dataframe(
        reduced_embeddings_3d_final, all_labels, all_positions, words_for_df_final
    )
    plot_dimensionality_reduction_results(
        df_plot_3d_final, words_to_test, OUTPUT_DIR, "Final_Token_Embeddings", "PCA"
    )
else:
    print("last_token_embeddings_list is empty. Skipping PCA and plotting for final hidden states.")
#%%
# --- Process and Plot FIRST Token Embeddings (Initial E_token + PE_pos) ---
if first_token_embeddings_list: # Assuming you have this list from the modified pipeline
    print("\n\n--- Processing FIRST Token Embeddings (Initial E_token + PE_pos) ---")
    embeddings_initial_concatenated = prepare_embeddings_for_pca(first_token_embeddings_list)
    
    # For 2D PCA Plot
    reduced_embeddings_2d_initial, first_token_eigenvectors = perform_dimensionality_reduction(
        embeddings_initial_concatenated, method="pca", n_components=2, random_state=RANDOM_STATE
    )
    words_for_df_initial = [label.split('_pos')[0] for label in all_labels] # Recalculate or pass
    df_plot_2d_initial = create_plot_dataframe(
        reduced_embeddings_2d_initial, all_labels, all_positions, words_for_df_initial
    )
    plot_dimensionality_reduction_results(
        df_plot_2d_initial, words_to_test, OUTPUT_DIR, "Initial_Frontend_Embeddings", "PCA"
    )

    # For 3D PCA Plot
    reduced_embeddings_3d_initial, first_token_eigenvectors3d = perform_dimensionality_reduction(
        embeddings_initial_concatenated, method="pca", n_components=3, random_state=RANDOM_STATE
    )
    df_plot_3d_initial = create_plot_dataframe(
        reduced_embeddings_3d_initial, all_labels, all_positions, words_for_df_initial
    )
    plot_dimensionality_reduction_results(
        df_plot_3d_initial, words_to_test, OUTPUT_DIR, "Initial_Frontend_Embeddings", "PCA"
    )
else:
    print("first_token_embeddings_list is empty. Skipping PCA and plotting for initial frontend embeddings.")

# %%

most_important_eigenvector=last_token_eigenvectors[0]

# Print the most important eigenvector
print("Most important eigenvector (first component):")
print(most_important_eigenvector)
# Plot the most important eigenvector
plt.figure(figsize=(10, 6))
plt.bar(range(len(most_important_eigenvector)), most_important_eigenvector)
plt.title("Most Important Eigenvector (First Component)")
plt.xlabel("Feature Index")
plt.ylabel("Eigenvector Value")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "most_important_eigenvector.png"))
plt.show()
# %%
# Example position
ex_idx = 100
example_token_sequence = all_token_sequences[ex_idx]
example_position = all_positions[ex_idx]

# Print the example token sequence and its position
print(f"Example token sequence at index {ex_idx}: {example_token_sequence}")
print(f"Example position for the token sequence: {example_position}")
# Print the corresponding word
example_word = all_labels[ex_idx].split('_pos')[0]

example_out = text2vec.predict_from_token_ids(
    example_token_sequence, # Add batch dimension
    target_device=DEVICE
)

example_sentence_embedding, example_last_hidden_states, example_initial_embeds, example_padding_mask = example_out

# Print the shapes of the outputs
print(f"Shape of example sentence embedding: {example_sentence_embedding.shape}")
print(f"Shape of example last hidden states: {example_last_hidden_states.shape}")
print(f"Shape of example initial embeds: {example_initial_embeds.shape}")

#%%
# %%

import src.sonar_encoder_decoder as sed
token2token = sed.SonarEncoderDecoder(
    device=DEVICE,
    model_name_encoder="text_sonar_basic_encoder",
    model_name_decoder="text_sonar_basic_decoder",  # Assuming you have a basic decoder
)
# %%

example_sentence = "The quick brown fox jumps over the lazy dog."
example_token_sequence = tokenizer_encoder(example_sentence)  # Get token IDs

example_out = text2vec.predict_from_token_ids(
    example_token_sequence, # Add batch dimension
    target_device=DEVICE,
    steering_vector=None,
    target_token_indices_for_steering=None  # Position of the target token
)

example_sentence_embedding, example_last_hidden_states, example_initial_embeds, example_padding_mask = example_out
example_sentence_embedding = example_sentence_embedding + torch.tensor(0.5*most_important_eigenvector).to(DEVICE).unsqueeze(0)  # Add the most important eigenvector
with torch.no_grad():
    example_decoded_tokens = token2token.decode_single(
        example_sentence_embedding, max_length=100
    )

print(token2token.token_ids_to_list_str(example_decoded_tokens))
# %%
embeddings_all_final_tokens = np.concatenate(last_token_embeddings_list, axis=0)
positions_array = np.array(all_positions.cpu(), dtype=int)
corrs = []
for i in range(embeddings_all_final_tokens.shape[1]):
    pearson_corr = np.corrcoef(
        embeddings_all_final_tokens[:, i], positions_array
    )
    corrs.append(pearson_corr[0, 1])  # Extract the correlation coefficient

plt.figure(figsize=(12, 6))
plt.plot(corrs, marker='o', linestyle='-', color='b')
plt.title("Pearson Correlation Coefficients of Final Token Embeddings vs. Positions")
plt.xlabel("Embedding Dimension Index")
plt.ylabel("Pearson Correlation Coefficient")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pearson_correlation_final_token_embeddings_vs_positions.png"))
plt.show()

# %%
