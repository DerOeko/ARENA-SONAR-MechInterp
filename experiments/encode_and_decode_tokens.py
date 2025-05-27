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
                # [
                #     ["dog", "dog"],
                #     ["house", "house"],
                #     ["car", "car"],
                #     ["cat", "cat"],
                #     ["dog", "house"],
                #     ["house", "dog"],
                # ]
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
import pandas as pd
from sklearn.decomposition import PCA

# Perform PCA on sentence embeddings
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(sentence_embeddings.detach().cpu().numpy())

# Create scatter plot with plotly
import plotly.express as px

# Create a DataFrame with the PCA components and labels
df = pd.DataFrame(
    {
        "PCA1": embeddings_2d[:, 0],
        "PCA2": embeddings_2d[:, 1],
        "Word": [
            "_ _ dog",
            "_ dog _",
            "dog _ _",
            "_ _ cat",
            "_ cat _",
            "cat _ _",
        ],  # Labels from the original input
    }
)

# Create interactive scatter plot
fig = px.scatter(
    df, x="PCA1", y="PCA2", text="Word", title="2D PCA Visualization of Word Embeddings"
)
fig.update_traces(textposition="top center")
fig.show()

# %%
dog_to_cat = sentence_embeddings[3] - sentence_embeddings[0]
cat_from_dog = sentence_embeddings[1] + dog_to_cat
# %%
# Create DataFrame with original points and the new transformed point
df_with_transform = pd.DataFrame(
    {
        "PCA1": np.append(
            embeddings_2d[:, 0],
            pca.transform(cat_from_dog.detach().cpu().numpy().reshape(1, -1))[0, 0],
        ),
        "PCA2": np.append(
            embeddings_2d[:, 1],
            pca.transform(cat_from_dog.detach().cpu().numpy().reshape(1, -1))[0, 1],
        ),
        "Word": [
            "_ _ dog",
            "_ dog _",
            "dog _ _",
            "_ _ cat",
            "_ cat _",
            "cat _ _",
            "cat from _ dog _",
        ],
    }
)

# Create interactive scatter plot with both original and transformed points
fig = px.scatter(
    df_with_transform,
    x="PCA1",
    y="PCA2",
    text="Word",
    title="2D PCA Visualization of Word Embeddings with Transformed Point",
)
fig.update_traces(textposition="top center")
fig.show()

# %%
encoder_decoder.token_ids_to_list_str_batch(encoder_decoder.decode(sentence_embeddings))

# %%
encoder_decoder.token_ids_to_list_str_batch(
    encoder_decoder.decode(cat_from_dog.unsqueeze(0))
)

# %%
sequence_embeddings = {}
for sequence in [
    ("cat",),
    ("dog",),
    ("_", "_", "dog"),
    ("_", "dog", "_"),
    ("dog", "_", "_"),
    ("_", "_", "cat"),
    ("_", "cat", "_"),
    ("cat", "_", "_"),
]:
    sentence_embeddings, encoded_seqs = encoder_decoder.encode(
        torch.cat(
            [
                encoder_decoder.list_str_to_token_ids_batch([sequence]),
            ]
        )
    )
    sequence_embeddings[sequence] = {
        "sequence_embeddings": sentence_embeddings,
        "decoded": encoder_decoder.token_ids_to_list_str_batch(
            encoder_decoder.decode(sentence_embeddings.unsqueeze(0))
        ),
    }
sequence_embeddings
# %%
dog_to_cat = (
    sequence_embeddings[("cat",)]["sequence_embeddings"]
    - sequence_embeddings[("dog",)]["sequence_embeddings"]
)
# %%
sequence_embeddings[("cat",)]["decoded"]

# %%
# Create transformed dog sequences by adding dog_to_cat vector
transformed_sequences = {}
for sequence, embedding_dict in sequence_embeddings.items():
    if "dog" in sequence:
        transformed_embedding = embedding_dict["sequence_embeddings"] + dog_to_cat
        transformed_sequences[sequence + ("transformed",)] = {
            "sequence_embeddings": transformed_embedding,
            "decoded": encoder_decoder.token_ids_to_list_str_batch(
                encoder_decoder.decode(transformed_embedding.unsqueeze(0))
            ),
        }
    # The cat sequences are already in sequence_embeddings and will be included
    # when we merge the dictionaries below with {**sequence_embeddings, **transformed_sequences}
    # We only need transformed_sequences to hold the transformed dog sequences

# Combine original and transformed embeddings
all_sequences = {**sequence_embeddings, **transformed_sequences}

# Prepare data for PCA
embeddings_matrix = torch.stack(
    [seq_dict["sequence_embeddings"] for seq_dict in all_sequences.values()]
)
sequence_labels = [" ".join(seq) for seq in all_sequences.keys()]

# Reshape embeddings matrix to 2D before PCA
embeddings_matrix_2d = embeddings_matrix.reshape(embeddings_matrix.shape[0], -1)

# Perform PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_matrix_2d.cpu().detach().numpy())

# Create DataFrame for plotting
df_all = pd.DataFrame(
    {
        "PCA1": embeddings_2d[:, 0],
        "PCA2": embeddings_2d[:, 1],
        "Sequence": sequence_labels,
    }
)

# Create scatter plot
fig = px.scatter(
    df_all,
    x="PCA1",
    y="PCA2",
    text="Sequence",
    title="2D PCA Visualization of Original and Transformed Sequences",
)
fig.update_traces(textposition="top center")
fig.show()

# %%
transformed_sequences
# %%
sequence_embeddings = {}
for sequence in [
    ("the", "dog", "was", "happy"),
    ("the", "dog", "was", "sad"),
    ("there", "was", "a", "dog"),
    ("dog", "is", "my", "name"),
    ("is", "the", "dog", "house"),
    ("is", "the", "dog", "happy"),
    ("the", "dog", "saw", "dog"),
    ("dog", "and", "dog", "played"),
    ("my", "dog", "likes", "dog"),
    ("dog", "sees", "another", "dog"),
    ("dog", "met", "the", "dog"),
    ("the", "cat", "saw", "dog"),
    ("cat", "and", "dog", "played"),
    ("my", "cat", "likes", "dog"),
    ("cat", "sees", "another", "dog"),
    ("dog", "met", "the", "cat"),
    ("dog", "dog", "dog", "dog"),
    ("cat", "cat", "cat", "cat"),
]:
    sentence_embeddings, encoded_seqs = encoder_decoder.encode(
        torch.cat(
            [
                encoder_decoder.list_str_to_token_ids_batch([sequence]),
            ]
        )
    )
    sequence_embeddings[sequence] = {
        "sequence_embeddings": sentence_embeddings,
        "decoded": encoder_decoder.token_ids_to_list_str_batch(
            encoder_decoder.decode(sentence_embeddings.unsqueeze(0))
        ),
    }
sequence_embeddings
# %%
# Get embeddings for original and transformed sequences
all_embeddings = []
sequence_labels = []

for sequence in sequence_embeddings.keys():
    # Original sequence embeddings
    orig_embedding = sequence_embeddings[sequence]["sequence_embeddings"]
    all_embeddings.append(orig_embedding.detach())
    sequence_labels.append(" ".join(sequence))

    # Transform sequence and get embeddings
    transformed_sequence = list(sequence)
    for i, word in enumerate(transformed_sequence):
        if word == "dog":
            transformed_sequence[i] = "cat"

    transformed_embeddings, _ = encoder_decoder.encode(
        torch.cat([encoder_decoder.list_str_to_token_ids_batch([transformed_sequence])])
    )
    all_embeddings.append(transformed_embeddings.detach())
    sequence_labels.append(" ".join(transformed_sequence))

# Stack all embeddings
embeddings_matrix = torch.stack(all_embeddings)

# Reshape embeddings matrix to 2D before PCA
embeddings_matrix_2d = embeddings_matrix.reshape(embeddings_matrix.shape[0], -1)

# Apply PCA
pca = PCA(n_components=3)  # Changed to 3 components
embeddings_3d = pca.fit_transform(embeddings_matrix_2d.cpu().numpy())

# Create DataFrame for plotting
df = pd.DataFrame(
    {
        "PCA1": embeddings_3d[:, 0],
        "PCA2": embeddings_3d[:, 1],
        "PCA3": embeddings_3d[:, 2],  # Added third component
        "Sequence": sequence_labels,
        "Type": [
            "Original" if i % 2 == 0 else "Transformed"
            for i in range(len(sequence_labels))
        ],
    }
)

# Create 3D scatter plot
fig = px.scatter_3d(
    df,
    x="PCA1",
    y="PCA2",
    z="PCA3",  # Added z dimension
    text="Sequence",
    color="Type",
    title="3D PCA Visualization of Original vs Dog->Cat Transformed Sequences",
)
fig.update_traces(textposition="top center")

# Enable zoom by updating layout
fig.update_layout(
    scene=dict(
        dragmode="orbit",
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5),
        ),
    )
)

fig.show()

# %%
# Decode the transformed embeddings back to text
decoded_sequences = []
for i in range(0, len(all_embeddings), 2):
    # Get original and transformed embeddings
    orig_emb = all_embeddings[i]
    trans_emb = all_embeddings[i + 1]

    # Decode both embeddings
    orig_decoded = encoder_decoder.decode(orig_emb.unsqueeze(0))
    trans_decoded = encoder_decoder.decode(trans_emb.unsqueeze(0))

    # Convert token IDs to strings
    orig_tokens = encoder_decoder.token_ids_to_list_str(orig_decoded[0])
    trans_tokens = encoder_decoder.token_ids_to_list_str(trans_decoded[0])

    # Add to list
    decoded_sequences.append({"Original": orig_tokens, "Transformed": trans_tokens})

# Print decoded sequences
print("\nDecoded Sequences:")
print("-" * 50)
for i, seq in enumerate(decoded_sequences):
    print(f"\nSequence pair {i + 1}:")
    print(f"Original:    {seq['Original']}")
    print(f"Transformed: {seq['Transformed']}")


# %%
encoder_decoder_spanish_to_spanish = SonarEncoderDecoder(
    device="cuda", decoder_language="spa_Latn"
)
# Decode the transformed embeddings back to text
decoded_sequences = []
for i in range(0, len(all_embeddings), 2):
    # Get original and transformed embeddings
    orig_emb = all_embeddings[i]
    trans_emb = all_embeddings[i + 1]

    # Decode both embeddings
    orig_decoded = encoder_decoder_spanish_to_spanish.decode(orig_emb.unsqueeze(0))
    trans_decoded = encoder_decoder_spanish_to_spanish.decode(trans_emb.unsqueeze(0))

    # Convert token IDs to strings
    orig_tokens = encoder_decoder_spanish_to_spanish.token_ids_to_list_str(
        orig_decoded[0]
    )
    trans_tokens = encoder_decoder_spanish_to_spanish.token_ids_to_list_str(
        trans_decoded[0]
    )

    # Add to list
    decoded_sequences.append({"Original": orig_tokens, "Transformed": trans_tokens})

# Print decoded sequences
print("\nDecoded Sequences:")
print("-" * 50)
for i, seq in enumerate(decoded_sequences):
    print(f"\nSequence pair {i + 1}:")
    print(f"Original:    {seq['Original']}")
    print(f"Transformed: {seq['Transformed']}")
# %%
# sequence_embeddings = {}
# for sequence in [
#     "thewolf",
#     "germanshepherd",
#     "goldenretriever",
#     "thefox",
# ]:
#     sentence_embeddings, encoded_seqs = encoder_decoder.encode(
#         torch.cat(
#             [
#                 encoder_decoder.list_str_to_token_ids_batch([sequence]),
#             ]
#         )
#     )
#     sequence_embeddings[sequence] = {
#         "sequence_embeddings": sentence_embeddings,
#         "decoded": encoder_decoder.token_ids_to_list_str_batch(
#             encoder_decoder.decode(sentence_embeddings.unsqueeze(0))
#         ),
#     }
# sequence_embeddings
# # %%
# # Apply transformation to each sequence
# print("\nApplying dog->cat transformation vector to sequences:")
# print("-" * 50)
# for sequence, data in sequence_embeddings.items():
#     # Get original embedding
#     orig_embedding = data["sequence_embeddings"]

#     # Apply transformation
#     transformed_embedding = orig_embedding + dog_to_cat

#     # Decode both original and transformed
#     orig_decoded = encoder_decoder.token_ids_to_list_str_batch(
#         encoder_decoder.decode(orig_embedding.unsqueeze(0))
#     )
#     trans_decoded = encoder_decoder.token_ids_to_list_str_batch(
#         encoder_decoder.decode(transformed_embedding.unsqueeze(0))
#     )

#     # print(f"\nOriginal sequence: {sequence}")
#     print(f"Original decoded:  {orig_decoded}")
#     print(f"Transformed decoded: {trans_decoded}")

# # %%
sequence_embeddings = {}
for sequence in [
    ("cat", "cat"),
    ("dog", "dog"),
    ("cat", "dog"),
    ("dog", "cat"),
]:
    sentence_embeddings, encoded_seqs = encoder_decoder.encode(
        torch.cat(
            [
                encoder_decoder.list_str_to_token_ids_batch([sequence]),
            ]
        )
    )
    sequence_embeddings[sequence] = {
        "sequence_embeddings": sentence_embeddings,
        "decoded": encoder_decoder.token_ids_to_list_str_batch(
            encoder_decoder.decode(sentence_embeddings.unsqueeze(0))
        ),
    }
sequence_embeddings
# # %%
dog_dog_to_cat_cat = (
    sequence_embeddings[("cat", "cat")]["sequence_embeddings"]
    - sequence_embeddings[("dog", "dog")]["sequence_embeddings"]
)
cat_dog_to_cat_cat = (
    sequence_embeddings[("cat", "cat")]["sequence_embeddings"]
    - sequence_embeddings[("cat", "dog")]["sequence_embeddings"]
)
dog_cat_to_cat_cat = (
    sequence_embeddings[("cat", "cat")]["sequence_embeddings"]
    - sequence_embeddings[("dog", "cat")]["sequence_embeddings"]
)
dog_dog_to_cat_dog = (
    sequence_embeddings[("cat", "dog")]["sequence_embeddings"]
    - sequence_embeddings[("dog", "dog")]["sequence_embeddings"]
)
dog_dog_to_dog_cat = (
    sequence_embeddings[("dog", "cat")]["sequence_embeddings"]
    - sequence_embeddings[("dog", "dog")]["sequence_embeddings"]
)
dog_cat_to_cat_dog = (
    sequence_embeddings[("cat", "dog")]["sequence_embeddings"]
    - sequence_embeddings[("dog", "cat")]["sequence_embeddings"]
)
cat_dog_to_dog_cat = (
    sequence_embeddings[("dog", "cat")]["sequence_embeddings"]
    - sequence_embeddings[("cat", "dog")]["sequence_embeddings"]
)


# %%

# Stack all vectors and perform PCA
all_vectors = torch.stack(
    [
        dog_dog_to_cat_cat,
        cat_dog_to_cat_cat,
        dog_cat_to_cat_cat,
        dog_dog_to_cat_dog,
        dog_dog_to_dog_cat,
        dog_cat_to_cat_dog,
        cat_dog_to_dog_cat,
    ]
)

# Reshape to 2D before PCA
all_vectors = all_vectors.view(all_vectors.size(0), -1)

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# Perform PCA
pca = PCA(n_components=3)
vectors_3d = pca.fit_transform(all_vectors.cpu().detach().numpy())

# Create figure
fig = go.Figure()

# Plot each vector as an arrow
vector_names = [
    "dog,dog→cat,cat",
    "cat,dog→cat,cat",
    "dog,cat→cat,cat",
    "dog,dog→cat,dog",
    "dog,dog→dog,cat",
    "dog,cat→cat,dog",
    "cat,dog→dog,cat",
]

colors = ["red", "blue", "green", "purple", "orange", "cyan", "magenta"]

for i, (vector, name, color) in enumerate(zip(vectors_3d, vector_names, colors)):
    # Add arrow
    fig.add_trace(
        go.Scatter3d(
            x=[0, vector[0]],
            y=[0, vector[1]],
            z=[0, vector[2]],
            mode="lines+text",
            line=dict(color=color, width=5),
            text=["", name],
            textposition="top center",
            name=name,
            showlegend=True,
        )
    )

# Update layout
fig.update_layout(
    scene=dict(
        xaxis_title="PC1",
        yaxis_title="PC2",
        zaxis_title="PC3",
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5),
        ),
    ),
    showlegend=True,
    title="PCA Visualization of Token Transformation Vectors",
)

fig.show()


# %%
# Create figure for positions
fig = go.Figure()

# Get embeddings for all combinations of cat and dog
sequences = [
    ("cat", "cat"),
    ("cat", "dog"),
    ("dog", "cat"),
    ("dog", "dog"),
    ("cat",),
    ("dog",),
]

positions = []
for seq in sequences:
    sentence_embeddings, _ = encoder_decoder.encode(
        torch.cat([encoder_decoder.list_str_to_token_ids_batch([seq])])
    )
    positions.append(sentence_embeddings.cpu().detach().numpy())

# Convert to numpy array and apply PCA
positions_array = np.vstack(positions)
pca = PCA(n_components=3)
positions_3d = pca.fit_transform(positions_array)

# Plot each position
colors = ["red", "blue", "green", "purple", "orange", "cyan"]

for i, (pos, seq, color) in enumerate(zip(positions_3d, sequences, colors)):
    # Add point
    name = ",".join(seq)
    fig.add_trace(
        go.Scatter3d(
            x=[pos[0]],
            y=[pos[1]],
            z=[pos[2]],
            mode="markers+text",
            marker=dict(size=10, color=color),
            text=[name],
            textposition="top center",
            name=name,
            showlegend=True,
        )
    )

# Update layout
fig.update_layout(
    scene=dict(
        xaxis_title="PC1",
        yaxis_title="PC2",
        zaxis_title="PC3",
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5),
        ),
    ),
    showlegend=True,
    title="PCA Visualization of Token Position Embeddings",
)

fig.show()

# %%
# Get vector from "cat" to "cat,cat"
# Need to first get embeddings for single "cat" token
sequence_embeddings_single = {}
for sequence in [("cat",)]:
    sentence_embeddings, encoded_seqs = encoder_decoder.encode(
        torch.cat(
            [
                encoder_decoder.list_str_to_token_ids_batch([sequence]),
            ]
        )
    )
    sequence_embeddings_single[sequence] = {
        "sequence_embeddings": sentence_embeddings,
        "decoded": encoder_decoder.token_ids_to_list_str_batch(
            encoder_decoder.decode(sentence_embeddings.unsqueeze(0))
        ),
    }

cat_to_cat_cat = (
    sequence_embeddings[("cat", "cat")]["sequence_embeddings"]
    - sequence_embeddings_single[("cat",)]["sequence_embeddings"]
)
cat_to_cat_cat

# %%
# Define a list of single tokens
single_tokens = [
    ("dog",),
    ("house",),
    ("tree",),
    ("car",),
    ("book",),
    ("bird",),
    ("fish",),
    ("table",),
    ("chair",),
    ("phone",),
    ("dog", "dog"),
    ("house", "house"),
    ("tree", "tree"),
    ("car", "car"),
    ("book", "book"),
    ("bird", "bird"),
    ("fish", "fish"),
    ("table", "table"),
    ("chair", "chair"),
    ("phone", "phone"),
    ("dog", "house"),
    ("dog", "bird"),
    ("house", "tree"),
    ("car", "book"),
    ("bird", "fish"),
    ("table", "chair"),
    ("phone", "car"),
    ("tree", "phone"),
    ("book", "table"),
    ("fish", "dog"),
]

# Get embeddings for each token
test_embeddings = {}
for sequence in single_tokens:
    sentence_embeddings, encoded_seqs = encoder_decoder.encode(
        torch.cat(
            [
                encoder_decoder.list_str_to_token_ids_batch([sequence]),
            ]
        )
    )
    test_embeddings[sequence] = {
        "sequence_embeddings": sentence_embeddings,
        "decoded": encoder_decoder.token_ids_to_list_str_batch(
            encoder_decoder.decode(sentence_embeddings.unsqueeze(0))
        ),
    }

# Apply the cat_to_cat_cat transformation and decode
print("\nApplying 'cat->cat,cat' transformation to single tokens:")
print("-" * 50)
# Store results in a list
results = []
for sequence, data in test_embeddings.items():
    # Get original embedding
    orig_embedding = data["sequence_embeddings"]

    # Apply transformation
    transformed_embedding = orig_embedding + cat_to_cat_cat

    # Decode both original and transformed
    orig_decoded = encoder_decoder.token_ids_to_list_str_batch(
        encoder_decoder.decode(orig_embedding.unsqueeze(0))
    )
    trans_decoded = encoder_decoder.token_ids_to_list_str_batch(
        encoder_decoder.decode(transformed_embedding.unsqueeze(0))
    )

    # Save results
    results.append(
        {"sequence": sequence, "original": orig_decoded, "transformed": trans_decoded}
    )

# Print all results
for result in results:
    print(f"\nOriginal token: {result['sequence']}")
    print(f"Original decoded: {result['original']}")
    print(f"Transformed decoded: {result['transformed']}")

# %%
