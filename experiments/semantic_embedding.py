# %%
import os
import random
import sys

import numpy as np
import torch

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
# Seed for reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(RANDOM_STATE)

# %%

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sonar_encoder_decoder import SonarEncoderDecoder

# %%
encoder_decoder = SonarEncoderDecoder(device="cuda")

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
