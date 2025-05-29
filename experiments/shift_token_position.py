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
                    ["_", "_", "dog"],
                    ["_", "dog", "_"],
                    ["dog", "_", "_"],
                    ["_", "_", "cat"],
                    ["_", "cat", "_"],
                    ["cat", "_", "_"],
                ]
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
sentence_embeddings
# %%

# Define sequences to test
sequences = [
    ("_", "_", "dog"),
    ("_", "dog", "_"),
    ("dog", "_", "_"),
    ("_", "_", "cat"),
    ("_", "cat", "_"),
    ("cat", "_", "_"),
]

# Dictionary to store results
sequence_data = {}

# Loop through each sequence
for sequence in sequences:
    # Convert sequence to tensor and encode
    sequence_tensor = encoder_decoder.list_str_to_token_ids_batch([list(sequence)])

    embedding, _ = encoder_decoder.encode(sequence_tensor)

    # Decode back to tokens
    decoded_ids = encoder_decoder.decode(embedding)
    decoded_tokens = encoder_decoder.token_ids_to_list_str_batch(decoded_ids)

    # Store in dictionary
    sequence_data[sequence] = {
        "embedding": embedding[0],  # Take first (only) embedding
        "decoded_tokens": decoded_tokens[0],  # Take first (only) decoded sequence
    }

# %%
sequence_data
# %%
dog_1_to_2 = (
    sequence_data[("_", "_", "dog")]["embedding"]
    - sequence_data[("_", "dog", "_")]["embedding"]
)
# %%
cat_shifted_1_to_2 = sequence_data[("_", "cat", "_")]["embedding"] + dog_1_to_2
# %%
encoder_decoder.token_ids_to_list_str_batch(
    encoder_decoder.decode(cat_shifted_1_to_2.unsqueeze(0))
)
# %%


def shift_token_position(sequence, shift_vector):
    sequence_tensor = encoder_decoder.list_str_to_token_ids_batch([list(sequence)])
    embedding, _ = encoder_decoder.encode(sequence_tensor)
    shifted = embedding + shift_vector
    return encoder_decoder.token_ids_to_list_str_batch(
        encoder_decoder.decode(shifted.unsqueeze(0))
    )


# %%
shift_token_position(("_", "cat", "_"), dog_1_to_2)
# %%
shift_token_position(("the", "cat", "the"), dog_1_to_2)
# %%
token_ids = list(range(encoder_decoder.tokenizer.model.vocabulary_size))


# %%
def find_shifting_vector(source_id_sequence, target_id_sequence):
    source_embedding, _ = encoder_decoder.encode(torch.tensor([source_id_sequence]))
    target_embedding, _ = encoder_decoder.encode(torch.tensor([target_id_sequence]))
    return target_embedding - source_embedding


# %%
# This is slow, and doesn't work. The idea was if we average the shifting the vector over a range of tokens
# and padding tokens, can we get a general vector to shift a token from one position to another?
# The answer seems to be no.


# source_position = 2
# target_position = 3

# def find_shifting_vector_averaged(source_position, target_position, n_tokens=10, n_padding_tokens=1000):
#     shifting_vectors = []
#     for token in token_ids[:n_tokens]:
#         for padding_token in token_ids[:n_padding_tokens]:
#             sequence = [padding_token] * (max(source_position, target_position) + 1)
#             source_sequence, target_sequence = sequence.copy(), sequence.copy()
#             source_sequence[source_position] = token
#             target_sequence[target_position] = token
#             shifting_vectors.append(find_shifting_vector(source_sequence, target_sequence))
#     return torch.stack(shifting_vectors).mean(dim=0)

# # %%
# shift_1_to_2 = find_shifting_vector_averaged(1, 2)
# # %%
# shift_token_from(("a", "house", "a"), shift_1_to_2)
# # %%
shift_token_position(
    ("dog", "house", "dog", "dog"),
    find_shifting_vector(
        source_id_sequence=encoder_decoder.list_str_to_token_ids(
            ("dog", "_", "_", "dog")
        ).tolist(),
        target_id_sequence=encoder_decoder.list_str_to_token_ids(
            ("cat", "_", "_", "dog")
        ).tolist(),
    ),
)

# %%


def modify_specific_token(sequence, token_position, new_token, filler_token="_"):
    source_token = sequence[token_position]
    sequence_cleaned = [
        filler_token if token != source_token else token for token in sequence
    ]
    target_sequence_cleaned = [
        token if i != token_position else new_token
        for i, token in enumerate(sequence_cleaned)
    ]
    shifting_vector = find_shifting_vector(
        source_id_sequence=encoder_decoder.list_str_to_token_ids(
            sequence_cleaned
        ).tolist(),
        target_id_sequence=encoder_decoder.list_str_to_token_ids(
            target_sequence_cleaned
        ).tolist(),
    )
    return shifting_vector, shift_token_position(sequence, shifting_vector)


# %%
outputs = []
sequence = ("dog", "a", "and", "dog", "b", "bit", "the", "other", "dog")
for i in range(len(sequence)):
    print(i)
    outputs.append(modify_specific_token(sequence, i, "house")[1])
outputs

# %%
shifting_vector, modified_sequence = modify_specific_token(sequence, 3, "house")
source_embedding = encoder_decoder.encode(
    encoder_decoder.list_str_to_token_ids(
        ("dog", "a", "and", "dog", "b", "bit", "the", "other", "dog")
    ).unsqueeze(0)
)[0]
target_embedding = encoder_decoder.encode(
    encoder_decoder.list_str_to_token_ids(
        ("dog", "a", "and", "house", "b", "bit", "the", "other", "dog")
    ).unsqueeze(0)
)[0]
print(f"{modified_sequence=}")
modified_embedding = source_embedding + shifting_vector
source_embedding, target_embedding, modified_embedding
# %%
import matplotlib.pyplot as plt

# Convert tensors to numpy arrays for plotting
source_np = source_embedding.numpy().flatten()
target_np = target_embedding.numpy().flatten()
modified_np = modified_embedding.numpy().flatten()

plt.figure(figsize=(15, 5))
plt.plot(source_np, label="Source Embedding", alpha=0.7)
plt.plot(target_np, label="Target Embedding", alpha=0.7)
plt.plot(modified_np, label="Modified Embedding", alpha=0.7)
plt.legend()
plt.title("Comparison of Embeddings")
plt.xlabel("Dimension")
plt.ylabel("Value")
plt.grid(True)
plt.show()


# %%
from torch.nn.functional import cosine_similarity

# Calculate cosine similarities between all pairs
source_target_sim = cosine_similarity(source_embedding, target_embedding)
source_modified_sim = cosine_similarity(source_embedding, modified_embedding)
target_modified_sim = cosine_similarity(target_embedding, modified_embedding)

print(f"Cosine similarity between source and target: {source_target_sim.item():.4f}")
print(
    f"Cosine similarity between source and modified: {source_modified_sim.item():.4f}"
)
print(
    f"Cosine similarity between target and modified: {target_modified_sim.item():.4f}"
)

# %%
# Calculate Euclidean distances between all pairs
source_target_dist = torch.norm(source_embedding - target_embedding)
source_modified_dist = torch.norm(source_embedding - modified_embedding)
target_modified_dist = torch.norm(target_embedding - modified_embedding)

print(f"Euclidean distance between source and target: {source_target_dist.item():.4f}")
print(
    f"Euclidean distance between source and modified: {source_modified_dist.item():.4f}"
)
print(
    f"Euclidean distance between target and modified: {target_modified_dist.item():.4f}"
)

# %%
# %%
difference_between_target_and_source_embedding = target_embedding - source_embedding
# Calculate cosine similarity between the difference vector and shifting vector
difference_shifting_sim = cosine_similarity(
    difference_between_target_and_source_embedding,
    shifting_vector,
)

print(
    f"Cosine similarity between difference vector and shifting vector: {difference_shifting_sim.item():.4f}"
)
source_target_dist = torch.norm(
    difference_between_target_and_source_embedding - shifting_vector
)
print(
    f"Euclidean distance between difference vector and shifting vector: {source_target_dist.item():.4f}"
)

# %%
plt.figure(figsize=(15, 5))
plt.plot(
    difference_between_target_and_source_embedding.detach().numpy().flatten(),
    label="Difference Vector",
    alpha=0.7,
)
plt.plot(shifting_vector.detach().numpy().flatten(), label="Shifting Vector", alpha=0.7)
plt.legend()
plt.title("Comparison of Difference and Shifting Vectors")
plt.xlabel("Dimension")
plt.ylabel("Value")
plt.grid(True)
plt.show()
# %%
plt.figure(figsize=(10, 10))
plt.scatter(
    difference_between_target_and_source_embedding.detach().numpy().flatten(),
    shifting_vector.detach().numpy().flatten(),
    alpha=0.5,
)
# plt.plot([-1, 1], [-1, 1], 'r--', alpha=0.3)  # Adding a diagonal reference line
plt.title("Difference Vector vs Shifting Vector\nComponent-wise Comparison")
plt.xlabel("Difference Vector Components")
plt.ylabel("Shifting Vector Components")
plt.grid(True)
plt.axis("equal")  # Make the plot square with equal scales
plt.show()

# %%
shifting_vector_pruned = shifting_vector.clone()
shifting_vector_pruned[
    (
        (shifting_vector_pruned - difference_between_target_and_source_embedding)
        / difference_between_target_and_source_embedding
    ).abs()
    > 0.7
] = 0.0
# %%
plt.figure(figsize=(15, 5))
plt.plot(
    difference_between_target_and_source_embedding.detach().numpy().flatten(),
    label="Difference Vector",
    alpha=0.7,
)
plt.plot(
    shifting_vector_pruned.detach().numpy().flatten(),
    label="Pruned Shifting Vector",
    alpha=0.7,
)
plt.legend()
plt.title("Comparison of Difference and Pruned Shifting Vectors")
plt.xlabel("Dimension")
plt.ylabel("Value")
plt.grid(True)
plt.show()
# %%
plt.figure(figsize=(10, 10))
plt.scatter(
    difference_between_target_and_source_embedding.detach().numpy().flatten(),
    shifting_vector_pruned.detach().numpy().flatten(),
    alpha=0.5,
)
# plt.plot([-1, 1], [-1, 1], 'r--', alpha=0.3)  # Adding a diagonal reference line
plt.title("Difference Vector vs Pruned Shifting Vector\nComponent-wise Comparison")
plt.xlabel("Difference Vector Components")
plt.ylabel("Pruned Shifting Vector Components")
plt.grid(True)
plt.axis("equal")  # Make the plot square with equal scales
plt.show()

# %%
encoder_decoder.token_ids_to_list_str_batch(
    encoder_decoder.decode(shifting_vector_pruned + source_embedding)
)

# %%
encoder_decoder.token_ids_to_list_str_batch(
    encoder_decoder.decode(shifting_vector + source_embedding)
)

# %%
from copy import copy

stats = {}
for source_sequence, token_position, new_token in [
    (
        ("dog", "a", "and", "dog", "b", "bit", "the", "other", "dog"),
        3,
        "house",
    ),
    (("a", "cat", "nap"), 1, "dog"),
    (("ant", "fly", "up"), 2, "down"),
    (("i", "ran", "far"), 1, "hop"),
    (("he", "run", "far"), 2, "near"),
    (("rat", "run", "low"), 0, "bat"),
    (("boy", "run", "out"), 2, "in"),
    (("air", "goes", "soft"), 1, "looks"),
    (("sun", "seems", "high"), 0, "moon"),
    (("rain", "fall", "light"), 1, "rise"),
]:
    target_sequence = list(copy(source_sequence))
    target_sequence[token_position] = new_token
    target_sequence = tuple(target_sequence)
    shifting_vector, modified_sequence = modify_specific_token(
        source_sequence, token_position, new_token
    )
    source_embedding = encoder_decoder.encode(
        encoder_decoder.list_str_to_token_ids(source_sequence).unsqueeze(0)
    )[0]
    target_embedding = encoder_decoder.encode(
        encoder_decoder.list_str_to_token_ids(target_sequence).unsqueeze(0)
    )[0]
    modified_embedding = source_embedding + shifting_vector
    difference_between_target_and_source_embedding = target_embedding - source_embedding
    found_pruned_shifting_vector = False
    for thresh in range(1, 10):
        try:
            thresh = i / 10
            shifting_vector_pruned = shifting_vector.clone()
            shifting_vector_pruned[
                (
                    (
                        shifting_vector_pruned
                        - difference_between_target_and_source_embedding
                    )
                    / difference_between_target_and_source_embedding
                ).abs()
                > thresh
            ] = 0.0
            pruned_sequence_output = tuple(
                encoder_decoder.token_ids_to_list_str_batch(
                    encoder_decoder.decode(shifting_vector_pruned + source_embedding)
                )[0]
            )[2:-1]
            pruned_sequence_output = tuple(
                token.strip("▁")
                for token in pruned_sequence_output  # note ▁ is not an underscore, all tokens seem prepended with it
            )
            print(pruned_sequence_output, target_sequence)
            # raise Exception("Stop here")
            if pruned_sequence_output == target_sequence:
                found_pruned_shifting_vector = True
                break
        except Exception as e:
            print(e)
            continue

    if found_pruned_shifting_vector:
        stats[(source_sequence, target_sequence)] = {
            "min_threshold": thresh,
            "shifting_vector_pruned": shifting_vector_pruned,
        }

stats
# %%
[stat["min_threshold"] for stat in stats.values()]
# %%
(
    torch.stack([stat["shifting_vector_pruned"] for stat in stats.values()], dim=0)
    == 0.0
).squeeze().all(dim=0).sum()
# %%
(
    torch.stack([stat["shifting_vector_pruned"] for stat in stats.values()], dim=0)
    == 0.0
).squeeze().sum(dim=1)
# %%

source_sequence = ("dog", "is", "in", "the", "house")
shifting_vector, modified_sequence = modify_specific_token(source_sequence, 0, "cat")
modified_sequence
# %%
gramatical_direction = (
    torch.from_numpy(np.load("./data/grammaticality_direction.npy"))
    .to(shifting_vector.device)
    .to(shifting_vector.dtype)
)
ungramatical_shifting_vector = shifting_vector - 0.0 * gramatical_direction
# %%
source_embedding = encoder_decoder.encode(
    encoder_decoder.list_str_to_token_ids(source_sequence).unsqueeze(0)
)[0]

encoder_decoder.token_ids_to_list_str_batch(
    encoder_decoder.decode(
        (source_embedding + ungramatical_shifting_vector).unsqueeze(0)
    )
)
# %%
# %%
source_sequence = ("dog", "house", "car", "died")

source_embedding = encoder_decoder.encode(
    encoder_decoder.list_str_to_token_ids(source_sequence).unsqueeze(0)
)[0]

res = {}
for gramaticality in [-10, -5, -1, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1, 5, 10]:
    source_embedding_modified = source_embedding + gramaticality * gramatical_direction

    res[gramaticality] = encoder_decoder.token_ids_to_list_str_batch(
        encoder_decoder.decode(source_embedding_modified.unsqueeze(0))
    )
res
# %%
