# %%
import torch
import random
import itertools # To help generate combinations
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fairseq2.typing import CPU, DataType, Device
from src.utils.pca_utils import (
    prepare_embeddings_for_pca,
    perform_dimensionality_reduction,
    create_plot_dataframe,
    plot_dimensionality_reduction_results
)

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
# Plotting
import plotly.express as px
import pandas as pd
from phate import PHATE
from src.sonar_encoder_decoder import SonarEncoderDecoder
from src.custom_sonar_pipeline import CustomTextToEmbeddingPipeline
import numpy as np

from utils.generate_random_sequences import generate_random_sequences
#%%
# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
MODEL_NAME = "text_sonar_basic_encoder"

OUTPUT_DIR = "../data/"

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
DOT_IDX = tokenizer_encoder(".")[1].item()  # Assuming '.' is a special token in the tokenizer
dummy_tokenized_for_special_tokens = tokenizer_encoder("test")
ENG_LANG_TOKEN_IDX = dummy_tokenized_for_special_tokens[0].item()

print(f"Using Language ID (eng_Latn): {ENG_LANG_TOKEN_IDX} ('{tokenizer_decoder(torch.tensor([ENG_LANG_TOKEN_IDX]))}')")
print(f"Using PAD ID: {PAD_IDX} ('{tokenizer_decoder(torch.tensor([PAD_IDX]))}')")
#%%
legal_code = open("../data/datasets/legal_code.txt", "r").read().split("\n")
num_legal_functions = len(legal_code)
print(f"Number of legal code examples: {num_legal_functions}")
print(f"First 5 function: {legal_code[:5]}")

illegal_code = open("../data/datasets/illegal_code.txt", "r").read().split("\n")
num_illegal_functions = len(illegal_code)
print(f"Number of illegal code examples: {num_illegal_functions}")
print(f"First 5 function: {illegal_code[:5]}")

grammatical_sentences = open("../data/datasets/grammatical_english_sentences.txt", "r").read().split("\n")
num_sentences = len(grammatical_sentences)
print(f"Number of grammatical sentences: {num_sentences}")
print(f"First 5 sentences: {grammatical_sentences[:5]}")

ungrammatical_sentences = open("../data/datasets/ungrammatical_english_sentences.txt", "r").read().split("\n")
num_ungrammatical_sentences = len(ungrammatical_sentences)
print(f"Number of ungrammatical sentences: {num_sentences}")
print(f"First 5 sentences: {ungrammatical_sentences[:5]}")

legal_lens = [len(seq) for seq in legal_code]
illegal_lens = [len(seq) for seq in illegal_code]
print(f"Average length of legal functions: {np.mean(legal_lens)}")
print(f"Average length of illegal functions: {np.mean(illegal_lens)}")
print(f"STD of legal functions: {np.std(legal_lens)}")
print(f"STD  of illegal functions: {np.std(illegal_lens)}")
#%% Get random sentences
random.seed(42)  # For reproducibility
print("--- Initializing Tokenizer and Special IDs ---")
orig_sonar_tokenizer = load_sonar_tokenizer("text_sonar_basic_encoder")
tokenizer_encoder = orig_sonar_tokenizer.create_encoder()
tokenizer_decoder = orig_sonar_tokenizer.create_decoder()

VOCAB_INFO = orig_sonar_tokenizer.vocab_info
vocab_size = VOCAB_INFO.size
print(f"Vocabulary size: {vocab_size}")

# Decide how many random sentences you want
desired_sequence_length = 20          # Total length of each random sequence
num_sentences = num_legal_functions + num_illegal_functions  # Total number of sequences to generate
grammatical_sentences = grammatical_sentences[:num_sentences]

print(f"\n--- Generating {num_sentences} Random Token Sequences ---")
random_sequences_batch = generate_random_sequences(
    num_sequences=num_sentences,
    seq_length=desired_sequence_length,
    vocab_size=vocab_size,
    end_token_id=DOT_IDX, # Using the DOT_IDX we defined
    device=DEVICE
)

print(f"Shape of generated random sequences batch: {random_sequences_batch.shape}")
if num_sentences > 0:
    print(f"Example of first 2 random sequences (token IDs):\n{random_sequences_batch[:2]}")
    print(f"Decoded first random sequence: '{tokenizer_decoder(random_sequences_batch[0])}'")

# %%
import tqdm
from tqdm.auto import tqdm
all_tokenized_sentences = []
for i in tqdm(range(0, len(grammatical_sentences), 1), desc="Tokenizing sentences"):
    batch_sentences = grammatical_sentences[i]
    tokenized = tokenizer_encoder(batch_sentences)
    
    # Convert to tensor and move to device
    tokenized_tensor = torch.tensor(tokenized, dtype=torch.long, device=DEVICE)
    all_tokenized_sentences.append(tokenized_tensor)
    
all_tokenized_legal_functions = []
for i in tqdm(range(0, len(legal_code), 1), desc="Tokenizing legal functions"):
    batch_sentences = legal_code[i]
    tokenized = tokenizer_encoder(batch_sentences)
    
    # Convert to tensor and move to device
    tokenized_tensor = torch.tensor(tokenized, dtype=torch.long, device=DEVICE)
    all_tokenized_legal_functions.append(tokenized_tensor)
    
all_tokenized_illegal_functions = []
for i in tqdm(range(0, len(illegal_code), 1), desc="Tokenizing illegal functions"):
    batch_sentences = illegal_code[i]
    tokenized = tokenizer_encoder(batch_sentences)
    
    # Convert to tensor and move to device
    tokenized_tensor = torch.tensor(tokenized, dtype=torch.long, device=DEVICE)
    all_tokenized_illegal_functions.append(tokenized_tensor)

max_len_legal = 0
for tokenized_func in all_tokenized_legal_functions:
    if tokenized_func.shape[0] > max_len_legal:
        max_len_legal = tokenized_func.shape[0]
print(f"Max sequence length for legal functions: {max_len_legal}")

max_len_illegal = 0
for tokenized_func in all_tokenized_illegal_functions:
    if tokenized_func.shape[0] > max_len_illegal:
        max_len_illegal = tokenized_func.shape[0]
print(f"Max sequence length for illegal functions: {max_len_illegal}")
# %%

for i in tqdm(range(0, len(all_tokenized_sentences), 1), desc="Padding sentences"):
    # Pad each sentence to the max length
    all_tokenized_sentences[i] = torch.nn.functional.pad(
        all_tokenized_sentences[i], 
        (0, desired_sequence_length - all_tokenized_sentences[i].shape[0]), 
        value=PAD_IDX
    )

for i in tqdm(range(0, len(all_tokenized_legal_functions), 1), desc="Padding legal functions"):
    # Pad each sentence to the max length
    all_tokenized_legal_functions[i] = torch.nn.functional.pad(
        all_tokenized_legal_functions[i], 
        (0, max_len_legal - all_tokenized_legal_functions[i].shape[0]), 
        value=PAD_IDX
    )

for i in tqdm(range(0, len(all_tokenized_illegal_functions), 1), desc="Padding illegal functions"):
    # Pad each sentence to the max length
    all_tokenized_illegal_functions[i] = torch.nn.functional.pad(
        all_tokenized_illegal_functions[i], 
        (0, max_len_illegal - all_tokenized_illegal_functions[i].shape[0]), 
        value=PAD_IDX
    )

    
# Convert the list of tensors to a single tensor
all_tokenized_sentences_tensor = torch.stack(all_tokenized_sentences)
print(f"Shape of all tokenized sentences tensor: {all_tokenized_sentences_tensor.shape}")

# Convert the list of tensors to a single tensor
all_tokenized_legal_tensor = torch.stack(all_tokenized_legal_functions)
print(f"Shape of all tokenized legal functions tensor: {all_tokenized_legal_tensor.shape}")

# Convert the list of tensors to a single tensor
all_tokenized_illegal_tensor = torch.stack(all_tokenized_illegal_functions)
print(f"Shape of all tokenized legal functions tensor: {all_tokenized_illegal_tensor.shape}")

# %% --- Custom Text to Embedding Pipeline ---
token2vec = CustomTextToEmbeddingPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=DEVICE
)

all_grammatical_sentence_embeddings = []
all_random_sentence_embeddings = []
all_legal_functions_embeddings = []
all_illegal_functions_embeddings = []
INFERENCE_BATCH_SIZE = 512

with torch.no_grad():
    for batch_start_idx in tqdm(range(0, len(all_tokenized_sentences_tensor), INFERENCE_BATCH_SIZE), desc="Generating Embeddings"):
        batch_end_idx = batch_start_idx + INFERENCE_BATCH_SIZE 
        #-- Grammatical Sentences Batch --#
        batch_tokens_ids = all_tokenized_sentences_tensor[batch_start_idx:batch_end_idx]

        if batch_tokens_ids.size(0) == 0:
            continue

        # Get sentence embeddings and all last hidden states for the batch
        batch_sentence_embeddings, batch_last_hidden_states, batch_first_hidden_states, _ = token2vec.predict_from_token_ids(batch_tokens_ids)
        
        # Store sentence embeddings (optional, if you still need them)
        all_grammatical_sentence_embeddings.append(batch_sentence_embeddings.cpu().numpy())

        #-- Random Sentences Batch --#
        batch_random_tokens_ids = random_sequences_batch[batch_start_idx:batch_end_idx]
        if batch_random_tokens_ids.size(0) == 0:
            continue
        
        # Get sentence embeddings and all last hidden states for the random batch
        batch_random_sentence_embeddings, batch_random_last_hidden_states, batch_random_first_hidden_states, _ = token2vec.predict_from_token_ids(batch_random_tokens_ids)
        
        # Store random sentence embeddings
        all_random_sentence_embeddings.append(batch_random_sentence_embeddings.cpu().numpy())
        
    for batch_start_idx in tqdm(range(0, len(all_tokenized_legal_tensor), INFERENCE_BATCH_SIZE), desc="Generating legal Function Embeddings"):
        batch_end_idx = batch_start_idx + INFERENCE_BATCH_SIZE 
        #-- legal Functions Batch --#
        batch_tokens_ids = all_tokenized_legal_tensor[batch_start_idx:batch_end_idx]

        if batch_tokens_ids.size(0) == 0:
            continue

        # Get sentence embeddings and all last hidden states for the batch
        batch_sentence_embeddings, batch_last_hidden_states, batch_first_hidden_states, _ = token2vec.predict_from_token_ids(batch_tokens_ids)
        
        # Store sentence embeddings (optional, if you still need them)
        all_legal_functions_embeddings.append(batch_sentence_embeddings.cpu().numpy())
    
    for batch_start_idx in tqdm(range(0, len(all_tokenized_illegal_tensor), INFERENCE_BATCH_SIZE), desc="Generating illegal Function Embeddings"):
        batch_end_idx = batch_start_idx + INFERENCE_BATCH_SIZE 
        #-- illegal Functions Batch --#
        batch_tokens_ids = all_tokenized_illegal_tensor[batch_start_idx:batch_end_idx]

        if batch_tokens_ids.size(0) == 0:
            continue

        # Get sentence embeddings and all last hidden states for the batch
        batch_sentence_embeddings, batch_last_hidden_states, batch_first_hidden_states, _ = token2vec.predict_from_token_ids(batch_tokens_ids)
        
        # Store sentence embeddings (optional, if you still need them)
        all_illegal_functions_embeddings.append(batch_sentence_embeddings.cpu().numpy())

all_grammatical_sentence_embeddings = np.concatenate(all_grammatical_sentence_embeddings, axis=0)
all_random_sentence_embeddings = np.concatenate(all_random_sentence_embeddings, axis=0)
all_legal_functions_embeddings = np.concatenate(all_legal_functions_embeddings, axis=0)
all_illegal_functions_embeddings = np.concatenate(all_illegal_functions_embeddings, axis=0)
print(f"Shape of all grammatical sentence embeddings: {all_grammatical_sentence_embeddings.shape}")
print(f"Shape of all random sentence embeddings: {all_random_sentence_embeddings.shape}")
print(f"Shape of all legal function embeddings: {all_legal_functions_embeddings.shape}")
print(f"Shape of all illegal function embeddings: {all_illegal_functions_embeddings.shape}")

# %%
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import numpy as np
import os
import torch

#labels_grammatical = ['grammatical'] * all_grammatical_sentence_embeddings.shape[0]
#labels_random = ['random'] * all_random_sentence_embeddings.shape[0]
labels_legal = ['legal'] * all_legal_functions_embeddings.shape[0]
labels_illegal = ['illegal'] * all_illegal_functions_embeddings.shape[0]
combined_lables = labels_legal + labels_illegal
#combined_lables = labels_random + labels_legal
combined_raw_embeddings = np.concatenate((all_legal_functions_embeddings, all_illegal_functions_embeddings), axis=0)
#combined_raw_embeddings = np.concatenate((all_random_sentence_embeddings, all_legal_functions_embeddings), axis=0)
print(f"Shape of combined raw embeddings for PCA: {combined_raw_embeddings.shape}")

#%% Perform PCA

operator = PCA(n_components=3, random_state=RANDOM_STATE)

reduced_embeddings = operator.fit_transform(combined_raw_embeddings)

eigenvectors = operator.components_

df = {
    'PCA1': reduced_embeddings[:, 0],
    'PCA2': reduced_embeddings[:, 1],
    'PCA3': reduced_embeddings[:, 2],
    'label': combined_lables
}

fig_interactive = px.scatter_3d(df, x='PCA1', y='PCA2', z='PCA3', color='label',
                                title='PCA of Sentence Embeddings',
                                labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2', 'PCA3': 'PCA Component 3'},
                                color_discrete_sequence=['blue', 'red', "green", "orange"])

# Add this to complete your analysis
fig_interactive.show()

# Also create a 2D version for clearer viewing
fig_2d = px.scatter(df, x='PCA1', y='PCA2', color='label',
                   title='PCA of Code Embeddings (2D)',
                   color_discrete_sequence=['blue', 'red'])
fig_2d.show()

# Calculate separation metrics
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(reduced_embeddings[:, :2], combined_lables)
print(f"Silhouette Score: {silhouette_avg}")
# %%
