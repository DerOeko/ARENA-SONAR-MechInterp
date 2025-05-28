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
english_encoder_decoder = SonarEncoderDecoder(
    device="cuda", encoder_language="eng_Latn", decoder_language="eng_Latn"
)
spanish_encoder_decoder = SonarEncoderDecoder(
    device="cuda", encoder_language="spa_Latn", decoder_language="spa_Latn"
)


# %%
english_sequence = ("dog",)
engligh_dog_embedding = english_encoder_decoder.encode(
    english_encoder_decoder.list_str_to_token_ids(english_sequence).unsqueeze(0)
)[0]
# %%
spanish_sequence = ("perro",)
spanish_dog_embedding = spanish_encoder_decoder.encode(
    spanish_encoder_decoder.list_str_to_token_ids(spanish_sequence).unsqueeze(0)
)[0]

# %%
engligh_dog_embedding - spanish_dog_embedding
# %%
# Calculate cosine similarity between English and Spanish embeddings
cosine_similarity = torch.nn.functional.cosine_similarity(
    engligh_dog_embedding, spanish_dog_embedding
)
print(
    f"Cosine similarity between 'dog' and 'perro' embeddings: {cosine_similarity.item():.4f}"
)

# %%
english_encoder_decoder.token_ids_to_list_str_batch(
    english_encoder_decoder.decode(engligh_dog_embedding.unsqueeze(0))
)
# %%
english_encoder_decoder.token_ids_to_list_str_batch(
    english_encoder_decoder.decode(spanish_dog_embedding.unsqueeze(0))
)
# %%
spanish_encoder_decoder.token_ids_to_list_str_batch(
    spanish_encoder_decoder.decode(engligh_dog_embedding.unsqueeze(0))
)
# %%
spanish_encoder_decoder.token_ids_to_list_str_batch(
    spanish_encoder_decoder.decode(spanish_dog_embedding.unsqueeze(0))
)
# %%
english_to_spanish_vector = spanish_dog_embedding - engligh_dog_embedding
english_to_spanish_vector
# %%
engligh_cat_embedding = english_encoder_decoder.encode(
    english_encoder_decoder.list_str_to_token_ids(("cat",)).unsqueeze(0)
)[0]
spanish_cat_embedding = spanish_encoder_decoder.encode(
    spanish_encoder_decoder.list_str_to_token_ids(("gato",)).unsqueeze(0)
)[0]

cosine_similarity = torch.nn.functional.cosine_similarity(
    spanish_cat_embedding, engligh_cat_embedding + english_to_spanish_vector
)
print(
    f"Cosine similarity between 'gato' and 'cat' transformed to spanish embeddings: {cosine_similarity.item():.4f}"
)
# %%

english_sentence = ("the", "dog", "and", "the", "cat")
spanish_sentence = ("el", "perro", "y", "el", "gato")

english_sentence_embedding = english_encoder_decoder.encode(
    english_encoder_decoder.list_str_to_token_ids(english_sentence).unsqueeze(0)
)[0]
spanish_sentence_embedding = spanish_encoder_decoder.encode(
    spanish_encoder_decoder.list_str_to_token_ids(spanish_sentence).unsqueeze(0)
)[0]

english_to_spanish_vector = spanish_sentence_embedding - english_sentence_embedding
english_to_spanish_vector
cosine_similarity = torch.nn.functional.cosine_similarity(
    spanish_encoder_decoder.encode(
        spanish_encoder_decoder.list_str_to_token_ids(
            ("la", "casa", "y", "el", "coche")
        ).unsqueeze(0)
    )[0],
    english_encoder_decoder.encode(
        english_encoder_decoder.list_str_to_token_ids(
            ("the", "house", "and", "the", "car")
        ).unsqueeze(0)
    )[0]
    + english_to_spanish_vector,
)
cosine_similarity

# %%
