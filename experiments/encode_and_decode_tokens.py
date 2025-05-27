# %%
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
# MODEL_NAME_ENCODER = "text_sonar_basic_encoder"
# MODEL_NAME_DECODER = "text_sonar_basic_decoder"

# Seed for reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(RANDOM_STATE)

# # %% --- Initialize Tokenizer and Model ---
# print("--- Initializing Tokenizer and Model ---")

# # Load tokenizer for special tokens
# tokenizer = load_sonar_tokenizer(MODEL_NAME_ENCODER)
# tokenizer_encoder = tokenizer.create_encoder()
# tokenizer_decoder = tokenizer.create_decoder()

# # Get special token IDs
# VOCAB_INFO = tokenizer.vocab_info
# PAD_IDX = VOCAB_INFO.pad_idx
# EOS_IDX = VOCAB_INFO.eos_idx
# UNK_IDX = VOCAB_INFO.unk_idx
# BOS_IDX = VOCAB_INFO.bos_idx

# # Get English language token ID
# dummy_tokenized = tokenizer_encoder("test")
# ENG_LANG_TOKEN_IDX = dummy_tokenized[0].item()

# # Load text-to-text model pipeline
# text_to_text_pipeline = TextToTextModelPipeline(
#     encoder=MODEL_NAME_ENCODER,
#     decoder=MODEL_NAME_DECODER,
#     tokenizer=MODEL_NAME_ENCODER,
#     device=DEVICE
# )


# print("Models initialized successfully.")

# # %%

# encoder = text_to_text_pipeline.model.encoder
# decoder = text_to_text_pipeline.model.decoder

# %%
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sonar_encoder_decoder import SonarEncoderDecoder

# %%
encoder_decoder = SonarEncoderDecoder(device="cuda")

# %%


# %%
# encoder_decoder.tokenizer.create_encoder()("hi")
# %%
unk_token_id = encoder_decoder.tokenizer.vocab_info.unk_idx
house_id = encoder_decoder.get_vocab_id("house")
dog_id = encoder_decoder.get_vocab_id("dog")
sentence_embeddings, encoded_seqs = encoder_decoder.encode(
    torch.cat(
        [
            encoder_decoder.list_str_to_token_ids_batch(
                [
                    ["I", "am", "a", "happy", "dog"],
                    ["I", "am", "a", "sad", "dog"],
                    ["a", "I", "dog", "happy", "am"],
                ]
            ),
            torch.tensor(
                [
                    [unk_token_id, unk_token_id, dog_id, unk_token_id, unk_token_id],
                    [dog_id, unk_token_id, unk_token_id, unk_token_id, unk_token_id],
                    [unk_token_id, unk_token_id, unk_token_id, unk_token_id, dog_id],
                    [unk_token_id, unk_token_id, house_id, unk_token_id, unk_token_id],
                    [house_id, unk_token_id, unk_token_id, unk_token_id, unk_token_id],
                    [unk_token_id, unk_token_id, unk_token_id, unk_token_id, house_id],
                ],
                device=DEVICE,
            ),
        ]
    )
)
sentence_embeddings, encoded_seqs
# %%

greedy_token_ids = encoder_decoder.decode(sentence_embeddings)
greedy_token_ids
# %%
encoder_decoder.token_ids_to_list_str_batch(greedy_token_ids)
# %%
encoder_decoder.decode(sentence_embeddings)

# %%
encoder_decoder.token_ids_to_list_str_batch(encoder_decoder.decode(sentence_embeddings))
# %%
