#%%
import os
import random
from typing import List

import numpy as np
import torch
from fairseq2.generation import BeamSearchSeq2SeqGenerator
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask
from fairseq2.typing import CPU, DataType, Device
from sonar.inference_pipelines.text import (
    EmbeddingToTextModelPipeline,
    TextToEmbeddingModelPipeline,
    TextToTextModelPipeline,
)
from sonar.models.sonar_text import load_sonar_tokenizer
from tqdm import tqdm

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
MODEL_NAME_ENCODER = "text_sonar_basic_encoder"
MODEL_NAME_DECODER = "text_sonar_basic_decoder"

# Seed for reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(RANDOM_STATE)

# %% --- Initialize Tokenizer and Model ---
print("--- Initializing Tokenizer and Model ---")

# Load tokenizer for special tokens
tokenizer = load_sonar_tokenizer(MODEL_NAME_ENCODER)
tokenizer_encoder = tokenizer.create_encoder()
tokenizer_decoder = tokenizer.create_decoder()

# Get special token IDs
VOCAB_INFO = tokenizer.vocab_info
PAD_IDX = VOCAB_INFO.pad_idx
EOS_IDX = VOCAB_INFO.eos_idx
UNK_IDX = VOCAB_INFO.unk_idx
BOS_IDX = VOCAB_INFO.bos_idx

# Get English language token ID
dummy_tokenized = tokenizer_encoder("test")
ENG_LANG_TOKEN_IDX = dummy_tokenized[0].item()

# Load text-to-text model pipeline
text_to_text_pipeline = TextToTextModelPipeline(
    encoder=MODEL_NAME_ENCODER,
    decoder=MODEL_NAME_DECODER, 
    tokenizer=MODEL_NAME_ENCODER,
    device=DEVICE
)


print("Models initialized successfully.")

# %%

encoder = text_to_text_pipeline.model.encoder
decoder = text_to_text_pipeline.model.decoder

# %%

def get_vocab_id(token: str) -> int:
    tokens = tokenizer_encoder(token)
    if len(tokens) != 3:
        raise RuntimeError("Multiple tokens found for token: " + token)
    return tokens[1].item()


# %%
house_idx = get_vocab_id("house")
# %%

# %% --- Test with a simple example ---
encoder_output = encoder.forward(SequenceBatch(
    torch.tensor([[ENG_LANG_TOKEN_IDX, get_vocab_id("dog"), PAD_IDX, PAD_IDX, EOS_IDX]]).to(DEVICE), None)
)
encoder_output


# Create initial token sequence with start token
start_tokens = torch.tensor([[BOS_IDX]]).to(DEVICE)

decoder_output = decoder.decode(
    seqs=start_tokens,
    padding_mask=None, 
    encoder_output=encoder_output.sentence_embeddings.unsqueeze(1),
    # encoder_output=encoder_output.encoded_seqs,
    encoder_padding_mask=None
)
decoder_output


# %%
greedy_token = decoder.project(decoder_output[0], decoder_padding_mask=None).logits.argmax()
greedy_token
# %%
def get_token_from_id(vocab_id: int) -> str:
    # The tokenizer's model has an index_to_token method
    return text_to_text_pipeline.tokenizer.model.index_to_token(vocab_id)

# Test it out with the house_idx
token = get_token_from_id(house_idx)
print(f"Token for index {house_idx}: {token}")

# %%
get_token_from_id(greedy_token)
# %%

seqs = torch.tensor([[ENG_LANG_TOKEN_IDX]]).to(DEVICE)

for i in range(10):
    decoder_output = decoder.decode(
        seqs=seqs,
        padding_mask=None, 
        # encoder_output=encoder_output.encoded_seqs,
        encoder_output=encoder_output.sentence_embeddings.unsqueeze(1),
        encoder_padding_mask=None
    )
    greedy_token = decoder.project(decoder_output[0], decoder_padding_mask=None).logits.argmax()
    seqs = torch.cat([seqs, greedy_token.unsqueeze(0).unsqueeze(0)], dim=1)
    print(seqs)
    print([get_token_from_id(token_id) for token_id in seqs[0]])
# %%
