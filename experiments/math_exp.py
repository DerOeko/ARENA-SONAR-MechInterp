#%%
import sys
import os
# Add the parent directory to the system path to import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.grammaticality_exp import generate_pca_plots_from_datasets

import torch
import random
import sys
import os
import torch
import numpy as np
import os
import torch
from src.utils.get_embeddings_similarity import calculate_embedding_list_similarity
from tqdm.auto import tqdm
from sonar.models.sonar_text import (
    load_sonar_tokenizer,
)
# SONAR and fairseq2 imports
from sonar.models.sonar_text import load_sonar_tokenizer
# Plotting
import plotly.express as px
from src.custom_sonar_pipeline import CustomTextToEmbeddingPipeline
import numpy as np

#%%
global DEVICE, RANDOM_STATE, MODEL_NAME, OUTPUT_DIR
# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
MODEL_NAME = "text_sonar_basic_encoder"

OUTPUT_DIR = "../data/"

#%%

num_samples = 1000

X_Y = []
Z = []
for _ in range(num_samples):
    # Generate a random math expression with 4 digits and basic operations
    digits = [random.randint(1000, 9999) for _ in range(3)]
    X_Y.append(f"{digits[0]} + {digits[1]}")

    Z.append(f"{digits[0] + digits[1] + random.randint(-1000, 1000)}")  # Adding some noise to the result


print(f"\n--- Calculating Embedding Similarity for X_Y vs Z ({num_samples} samples) ---")
try:
    similarity_results = calculate_embedding_list_similarity(
        dataset1_texts=X_Y,
        dataset2_texts=Z,
        model_name=MODEL_NAME,        # Using your global MODEL_NAME
        device_override=DEVICE,       # Using your global DEVICE
        random_seed=RANDOM_STATE,     # Using your global RANDOM_STATE
        # You can also set other parameters like:
        # inference_batch_size=128, # (default is often 128)
        # return_individual_dataset2_similarities=True, # (default is True)
        # return_average_embeddings_data=False # (default is False)
    )

    print("\nSimilarity Analysis Results:")
    print(f"  Overall similarity [Avg(X_Y) vs Avg(Z)]: {similarity_results['similarity_avg_dataset1_vs_avg_dataset2']:.4f}")
    print(f"  Number of embeddings in X_Y: {similarity_results['num_embeddings_dataset1']}")
    print(f"  Number of embeddings in Z: {similarity_results['num_embeddings_dataset2']}")
    print(f"  Uniform padding length used: {similarity_results['uniform_padding_length_used']}")

    if "avg_similarity_of_individual_dataset2_vs_avg_dataset1" in similarity_results:
        print(f"  Average of individual Z sentence similarities to Avg(X_Y): {similarity_results['avg_similarity_of_individual_dataset2_vs_avg_dataset1']:.4f}")

    # Individual similarities (if returned):
    # if similarity_results.get("individual_dataset2_similarities_vs_avg_dataset1"):
    #     print("\n  First 3 individual Z similarities to Avg(X_Y):")
    #     for item in similarity_results["individual_dataset2_similarities_vs_avg_dataset1"][:3]:
    #         print(f"    Text: \"{item['text_dataset2']}\", Similarity: {item['similarity_to_avg_datase        t1']:.4f}")

except Exception as e:
    print(f"An error occurred during the similarity calculation: {e}")

# %%

#%% generate datasets, one is true, one is false
num_samples = 1000

incorrect_math_sequences = []
correct_math_sequences = []

# Define operations and their corresponding functions
operations = {
    '+': lambda a, b: a + b,
    #'-': lambda a, b: a - b,
    #'*': lambda a, b: a * b
}

for _ in range(num_samples):
    # Generate a random math expression with 4 digits and basic operations
    len_of_digits = 4  # Fixed at 4 digits
    
    # Randomly select an operation (only once!)
    op_symbol = random.choice(list(operations.keys()))
    operation_func = operations[op_symbol]
    
    # Generate first two digits normally
    a = random.randint(int("1" + "0" * len_of_digits), int("1" + "0" * (len_of_digits+1)) - 1)
    b = random.randint(int("1" + "0" * len_of_digits), int("1" + "0" * (len_of_digits+1)) - 1)
    
    # Calculate the correct result
    correct_result = operation_func(a, b)
    
    # For subtraction, ensure positive result
    if op_symbol == '-' and a < b:
        a, b = b, a  # Swap to ensure positive result
        correct_result = operation_func(a, b)
    
    # Generate wrong answer with similar magnitude
    if op_symbol == '*':
        # For multiplication, generate a wrong answer with similar number of digits
        result_magnitude = len(str(correct_result))
        wrong_result = random.randint(10**(result_magnitude-1), 10**result_magnitude - 1)
        
        # Ensure the wrong result is actually different
        while wrong_result == correct_result:
            wrong_result = random.randint(10**(result_magnitude-1), 10**result_magnitude - 1)
    else:
        # For addition/subtraction, just use a random number with similar digits
        wrong_result = random.randint(int("1" + "0" * len_of_digits), int("1" + "0" * (len_of_digits+1)) - 1)
        
        # Ensure the wrong result is actually different
        while wrong_result == correct_result:
            wrong_result = random.randint(int("1" + "0" * len_of_digits), int("1" + "0" * (len_of_digits+1)) - 1)
    
    # Create text sequences
    correct_math_sequences.append(f"The result of {a} {op_symbol} {b} is {correct_result}")
    incorrect_math_sequences.append(f"The result of {a} {op_symbol} {b} is {wrong_result}")

datasets = [correct_math_sequences, incorrect_math_sequences]
labels = ["correct_math_sequences", "incorrect_math_sequences"]
#%% tokenize both
# --- Custom Text to Embedding Pipeline & Model Max Sequence Length ---
token2vec = CustomTextToEmbeddingPipeline(
    encoder=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=DEVICE,
)

max_seq_len = token2vec.model.encoder_frontend.pos_encoder.max_seq_len

# Seed for reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(RANDOM_STATE)

orig_sonar_tokenizer = load_sonar_tokenizer(MODEL_NAME)
tokenizer_encoder = orig_sonar_tokenizer.create_encoder()

VOCAB_INFO = orig_sonar_tokenizer.vocab_info
PAD_IDX = VOCAB_INFO.pad_idx

dataset_texts = {label: [] for label in labels}
max_len_overall = 0

print("--- Loading and Tokenizing Datasets ---")

if datasets is not None:
    for i in range(len(datasets)):
        texts = datasets[i]
        label = labels[i]
        
        print(f"Processing {label} dataset ({len(texts)} sentences)...")
        tokenized_sentences_for_label = []
        filtered_texts_for_label = []  # NEW: Keep track of filtered texts
        filtered_count = 0
        
        for text in tqdm(texts, desc=f"Tokenizing {label}"):
            tokenized = tokenizer_encoder(text)
            tokenized_tensor = torch.tensor(tokenized, dtype=torch.long, device=DEVICE)
            
            # Filter out sequences that are too long
            if tokenized_tensor.shape[0] <= max_seq_len:
                tokenized_sentences_for_label.append(tokenized_tensor)
                filtered_texts_for_label.append(text)  # Keep corresponding text
                if tokenized_tensor.shape[0] > max_len_overall:
                    max_len_overall = tokenized_tensor.shape[0]
            else:
                filtered_count += 1
                
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} sequences longer than {max_seq_len} tokens for {label}")
            
        dataset_texts[label] = {"raw_texts": filtered_texts_for_label, "tokenized_tensors": tokenized_sentences_for_label}
else:
    raise ValueError("Either datasets_paths or datasets must be provided.")

max_len_overall = min(max_len_overall, max_seq_len)  # Ensure we don't exceed max_seq_len


# --- Padding ---
tokenized_padded_tensors = {}
for label, data in dataset_texts.items():
    print(f"Padding {label} dataset...")
    padded_tensors = []
    for tokenized_tensor in tqdm(data["tokenized_tensors"], desc=f"Padding {label}"):
        padded_tensor = torch.nn.functional.pad(
            tokenized_tensor,
            (0, max_len_overall - tokenized_tensor.shape[0]),
            value=PAD_IDX
        )
        padded_tensors.append(padded_tensor)
    tokenized_padded_tensors[label] = torch.stack(padded_tensors)
    print(f"Shape of padded tokenized tensor for {label}: {tokenized_padded_tensors[label].shape}")


#%% Generate embeddings
all_embeddings = {label: [] for label in labels}
INFERENCE_BATCH_SIZE = 128

print("--- Generating Embeddings ---")
for label, tokenized_tensor in tokenized_padded_tensors.items():
    print(f"Generating embeddings for {label} dataset...")
    with torch.no_grad():
        for batch_start_idx in tqdm(range(0, len(tokenized_tensor), INFERENCE_BATCH_SIZE), desc=f"Embedding {label}"):
            batch_end_idx = batch_start_idx + INFERENCE_BATCH_SIZE
            batch_tokens_ids = tokenized_tensor[batch_start_idx:batch_end_idx]

            if batch_tokens_ids.size(0) == 0:
                continue

            batch_sentence_embeddings, _, _, _ = token2vec.predict_from_token_ids(batch_tokens_ids)
            all_embeddings[label].append(batch_sentence_embeddings.cpu().numpy())

    all_embeddings[label] = np.concatenate(all_embeddings[label], axis=0)
    print(f"Shape of {label} embeddings: {all_embeddings[label].shape}")

# --- Prepare for Dimensionality Reduction ---
labels_vec = []
combined_raw_embeddings = []
combined_raw_texts = [] # Store raw texts for plotting
for label in labels:
    embeddings = all_embeddings[label]
    labels_vec.extend([label] * embeddings.shape[0])
    combined_raw_embeddings.append(embeddings)
    combined_raw_texts.extend(dataset_texts[label]["raw_texts"]) # Add raw texts

combined_raw_embeddings = np.concatenate(combined_raw_embeddings, axis=0)
print(f"Shape of combined raw embeddings: {combined_raw_embeddings.shape}")
# %% calculate mean OVERALL embedding
mean_embeddings = np.mean(combined_raw_embeddings, axis=0)

# Center the embeddings by subtracting the mean
centered_embeddings = combined_raw_embeddings - mean_embeddings

#%% Learn the General Mathematical Correctness Direction
tau_vec = np.array([1.0 if label == "correct_math_sequences" else -1.0 for label in labels_vec])

tau_vec.shape
#%%
t_math_g = np.mean(tau_vec[:, np.newaxis] * centered_embeddings, axis=0)

print(f"Shape of t_math_g (Mathematical Correctness Direction): {t_math_g.shape}")
print(f"First 5 elements of t_math_g: {t_math_g[:5]}")
# %% get squareroot of num of tokens
squared_tokens = []
for label in labels:
    dataset_texts[label]['num_tokens'] = [
        len(token_tensor) for token_tensor in dataset_texts[label]["tokenized_tensors"]
    ]
    squared_tokens.extend(np.sqrt(np.array(dataset_texts[label]['num_tokens'])))
# %%

proj_math_g = np.dot(centered_embeddings, t_math_g[:, np.newaxis]) * np.array(squared_tokens)[:, np.newaxis]

preds = [-1 if proj_math_g[i] < 0 else 1 for i in range(len(proj_math_g))]
# %%

acc = np.mean(np.array(preds) == tau_vec)
print(f"Accuracy of predictions: {acc:.4f}")

# %%
from numpy.linalg import norm

datasets = [correct_math_sequences, incorrect_math_sequences]
labels = ["correct_math_sequences", "incorrect_math_sequences"]

reduced_embeddings, df, fig_interactive, eigenvectors, grammaticality_direction = generate_pca_plots_from_datasets(datasets=datasets, labels=labels, n_components=2, reduction_method="PCA", return_eigenvectors=True, enable_correctness_direction_analysis=True)

#%% calculate cosinge similarity between grammaticality and correctness direction
dot_product_ab = np.dot(grammaticality_direction[np.newaxis, :], t_math_g[:, np.newaxis])

norm_t_math_g = norm(t_math_g)

# Calculate cosine similarity
# Ensure norm_t_math_g is not zero to avoid division by zero
if norm_t_math_g == 0:
    cosine_similarity_val = 0 # Or handle as NaN, or error
else:
    cosine_similarity_val = dot_product_ab / norm_t_math_g

print(f"Cosine Similarity (corrected): {cosine_similarity_val}")
# %%
