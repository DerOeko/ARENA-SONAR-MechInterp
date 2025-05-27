# %%
import torch
import random
import itertools # To help generate combinations
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fairseq2.typing import CPU, DataType, Device
from utils.pca_utils import (
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
grammatical_sentences = open("../data/generated_sentences.txt", "r").read().split("\n")
num_sentences = len(grammatical_sentences)
print(f"Number of grammatical sentences: {num_sentences}")
print(f"First 5 sentences: {grammatical_sentences[:5]}")

#%% Get random sentences
random.seed(42)  # For reproducibility
print("--- Initializing Tokenizer and Special IDs ---")
orig_sonar_tokenizer = load_sonar_tokenizer("text_sonar_basic_encoder")
tokenizer_encoder = orig_sonar_tokenizer.create_encoder()
tokenizer_decoder = orig_sonar_tokenizer.create_decoder()

VOCAB_INFO = orig_sonar_tokenizer.vocab_info
vocab_size = VOCAB_INFO.size
print(f"Vocabulary size: {vocab_size}")

def generate_random_sequences(
    num_sequences: int,
    vocab_size: int,
    seq_length: int,
    end_token_id: int,
    device: torch.device
) -> torch.Tensor:
    """
    Generates a batch of random token sequences, each of `seq_length`,
    with the last token being `end_token_id`.

    Args:
        num_sequences: Number of random sequences to generate.
        seq_length: The total length of each sequence (including the end_token_id).
        vocab_size: The size of the vocabulary to sample from.
        end_token_id: The token ID to place at the end of each sequence.
        device: The torch device to create tensors on.

    Returns:
        A Tensor of shape [num_sequences, seq_length] with random token IDs.
    """    
    if seq_length == 1:
        # If sequence length is 1, it's just the end token
        sequences = torch.full((num_sequences, 1), fill_value=end_token_id, dtype=torch.long, device=device)
    else:
        # Generate (seq_length - 1) random tokens for each sequence
        random_parts = torch.randint(0, vocab_size, (num_sequences, seq_length - 1), device=device)
        
        # Create a tensor for the end_token_id, repeated for each sequence
        end_tokens = torch.full((num_sequences, 1), fill_value=end_token_id, dtype=torch.long, device=device)
        
        # Concatenate the random parts with the end_token_id
        sequences = torch.cat((random_parts, end_tokens), dim=1)
        
    return sequences
# %%

# Get vocab_size from your VOCAB_INFO
vocab_size = VOCAB_INFO.size # Or VOCAB_INFO.vocab_size, check the attribute name

# Decide how many random sentences you want
desired_sequence_length = 20          # Total length of each random sequence

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

# %%

for i in tqdm(range(0, len(all_tokenized_sentences), 1), desc="Padding sentences"):
    # Pad each sentence to the max length
    all_tokenized_sentences[i] = torch.nn.functional.pad(
        all_tokenized_sentences[i], 
        (0, desired_sequence_length - all_tokenized_sentences[i].shape[0]), 
        value=PAD_IDX
    )
    
# Convert the list of tensors to a single tensor
all_tokenized_sentences_tensor = torch.stack(all_tokenized_sentences)
print(f"Shape of all tokenized sentences tensor: {all_tokenized_sentences_tensor.shape}")

# %% --- Custom Text to Embedding Pipeline ---
token2vec = CustomTextToEmbeddingPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=DEVICE
)

all_grammatical_sentence_embeddings = []
all_random_sentence_embeddings = []
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

all_grammatical_sentence_embeddings = np.concatenate(all_grammatical_sentence_embeddings, axis=0)
all_random_sentence_embeddings = np.concatenate(all_random_sentence_embeddings, axis=0)
print(f"Shape of all grammatical sentence embeddings: {all_grammatical_sentence_embeddings.shape}")
print(f"Shape of all random sentence embeddings: {all_random_sentence_embeddings.shape}")
# %%
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import numpy as np
import os
import torch

labels_grammatical = ['grammatical'] * all_grammatical_sentence_embeddings.shape[0]
labels_random = ['random'] * all_random_sentence_embeddings.shape[0]

combined_lables = labels_grammatical + labels_random

combined_raw_embeddings = np.concatenate((all_grammatical_sentence_embeddings, all_random_sentence_embeddings), axis=0)

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
                                color_discrete_sequence=['blue', 'red'])
#%%
grammar_embeddings, grammar_eigenvectors = perform_dimensionality_reduction(
    embeddings=all_grammatical_sentence_embeddings,
    method="pca",
    n_components=2,
    random_state=RANDOM_STATE
)

random_embeddings, random_eigenvectors = perform_dimensionality_reduction(
    embeddings=all_random_sentence_embeddings,
    method="pca",
    n_components=2,
    random_state=RANDOM_STATE
)
# %%

df_grammar = create_plot_dataframe()
# %%

# %%
