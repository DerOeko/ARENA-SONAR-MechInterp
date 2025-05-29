# %%
# Ensure all necessary imports from your original script are here
import torch
import random
import sys
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity # For similarity calculation
from tqdm.auto import tqdm

# Assuming these paths and custom modules are correctly set up as in your environment.
# You'll need to ensure CustomTextToEmbeddingPipeline and load_sonar_tokenizer are importable.
# Example:
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Adjust if needed
from sonar.models.sonar_text import (
    load_sonar_tokenizer,
)
from src.custom_sonar_pipeline import CustomTextToEmbeddingPipeline

# %%

# Helper function to tokenize, pad, and generate embeddings for a list of texts
# This helper function remains largely the same as in the previous more complex version.
def _get_embeddings_for_texts_core(
    texts_list: list[str],
    processing_label: str, # For progress bar descriptions
    token2vec_pipeline: CustomTextToEmbeddingPipeline,
    tokenizer_encoder_callable, # The callable encoder from SONAR tokenizer
    model_max_allowable_seq_len: int,
    vocab_pad_idx: int,
    target_pad_length: int, # The uniform length all sequences should be padded to
    processing_device: torch.device,
    batch_size_for_inference: int = 128
) -> tuple[np.ndarray, list[str]]:
    """
    Tokenizes, pads, and generates embeddings for a list of texts.
    Returns a tuple of (numpy array of embeddings, list of filtered raw texts that were embedded).
    """
    if not texts_list:
        print(f"Warning: Empty text list provided for {processing_label}.")
        return np.array([]), []

    tokenized_tensors_list = []
    retained_raw_texts = []
    num_filtered_due_to_length = 0

    for text_item in tqdm(texts_list, desc=f"Tokenizing {processing_label}"):
        token_ids = tokenizer_encoder_callable(text_item)
        if 0 < len(token_ids) <= model_max_allowable_seq_len: # Ensure non-empty and within limit
            tokenized_tensors_list.append(torch.tensor(token_ids, dtype=torch.long))
            retained_raw_texts.append(text_item)
        elif len(token_ids) > 0: # Only count if it wasn't an empty tokenization result
            num_filtered_due_to_length += 1
    
    if num_filtered_due_to_length > 0:
        print(f"Filtered out {num_filtered_due_to_length} sequences longer than {model_max_allowable_seq_len} tokens for {processing_label}.")

    if not tokenized_tensors_list:
        print(f"No sequences remaining for {processing_label} after length filtering.")
        return np.array([]), []

    padded_tensors_for_batching = []
    for token_tensor in tqdm(tokenized_tensors_list, desc=f"Padding {processing_label}"):
        pad_amount = target_pad_length - token_tensor.shape[0]
        if pad_amount < 0:
            # This case should ideally be prevented by how target_pad_length is determined
            print(f"Warning: Sequence in {processing_label} is longer ({token_tensor.shape[0]}) than target_pad_length ({target_pad_length}). Truncating.")
            token_tensor = token_tensor[:target_pad_length]
            pad_amount = 0
        
        padded_tensor = torch.nn.functional.pad(
            token_tensor, (0, pad_amount), value=vocab_pad_idx
        )
        padded_tensors_for_batching.append(padded_tensor)
    
    if not padded_tensors_for_batching: # Should be redundant if tokenized_tensors_list was not empty
        return np.array([]), retained_raw_texts

    stacked_token_ids_tensor = torch.stack(padded_tensors_for_batching).to(processing_device)

    all_batch_embeddings_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(stacked_token_ids_tensor), batch_size_for_inference), desc=f"Embedding {processing_label}"):
            batch_token_ids = stacked_token_ids_tensor[i : i + batch_size_for_inference]
            if batch_token_ids.size(0) == 0:
                continue
            
            batch_sentence_embeddings, _, _, _ = token2vec_pipeline.predict_from_token_ids(batch_token_ids)
            all_batch_embeddings_list.append(batch_sentence_embeddings.cpu().numpy())
    
    if not all_batch_embeddings_list:
        print(f"No embeddings generated for {processing_label}, possibly due to empty batches.")
        return np.array([]), retained_raw_texts
        
    final_embeddings_array = np.concatenate(all_batch_embeddings_list, axis=0)
    return final_embeddings_array, retained_raw_texts


def calculate_embedding_list_similarity(
    dataset1_texts: list[str],
    dataset2_texts: list[str],
    model_name: str = "text_sonar_basic_encoder",
    device_override: torch.device = None,
    random_seed: int = 42,
    inference_batch_size: int = 128,
    return_individual_dataset2_similarities: bool = True,
    return_average_embeddings_data: bool = False
) -> dict:
    """
    Calculates cosine similarity between the average embedding of dataset1_texts
    and the average embedding of dataset2_texts.

    Args:
        dataset1_texts: A list of strings for the first dataset.
        dataset2_texts: A list of strings for the second dataset.
        model_name: Name of the SONAR model.
        device_override: Specific torch device to use. Auto-detects if None.
        random_seed: Seed for reproducibility.
        inference_batch_size: Batch size for embedding.
        return_individual_dataset2_similarities: If True, returns similarities of each
                                                 sentence in dataset2 to the average
                                                 embedding of dataset1.
        return_average_embeddings_data: If True, includes the computed average embeddings.

    Returns:
        A dictionary with similarity scores and other optional data.
    """

    # --- 1. Setup and Initialization ---
    print("--- Initializing Configurations and Model ---")
    selected_device = device_override if device_override else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(random_seed)

    token2vec = CustomTextToEmbeddingPipeline(
        encoder=model_name, tokenizer=model_name, device=selected_device,
    )
    model_max_seq_len = token2vec.model.encoder_frontend.pos_encoder.max_seq_len
    
    sonar_tokenizer_instance = load_sonar_tokenizer(model_name)
    tokenizer_encoder = sonar_tokenizer_instance.create_encoder()
    pad_idx = sonar_tokenizer_instance.vocab_info.pad_idx

    print(f"Using Device: {selected_device}, Model: {model_name}, Model Max Seq Len: {model_max_seq_len}")

    # --- 2. Validate Input Data ---
    if not dataset1_texts:
        raise ValueError("Dataset 1 texts list is empty.")
    if not dataset2_texts:
        raise ValueError("Dataset 2 texts list is empty.")
    print(f"Dataset 1: {len(dataset1_texts)} texts. Dataset 2: {len(dataset2_texts)} texts.")

    # --- 3. Determine Uniform Padding Length ---
    print("--- Determining Uniform Padding Length ---")
    max_observed_token_len = 0
    all_texts_for_length_check = dataset1_texts + dataset2_texts
    
    if not all_texts_for_length_check: # Should not happen if previous checks pass
        raise ValueError("Combined list for length check is empty.")

    for text_item in tqdm(all_texts_for_length_check, desc="Pre-calculating max token length"):
        token_ids = tokenizer_encoder(text_item)
        if 0 < len(token_ids) <= model_max_seq_len:
            if len(token_ids) > max_observed_token_len:
                max_observed_token_len = len(token_ids)
    
    if max_observed_token_len == 0:
        raise ValueError("No valid sequences (within model's max length and non-empty) found. Cannot determine padding length.")
    
    uniform_padding_length = max_observed_token_len
    print(f"Uniform padding length for all sequences set to: {uniform_padding_length}")

    # --- 4. Generate Embeddings ---
    embeddings_d1, filtered_texts_d1 = _get_embeddings_for_texts_core(
        dataset1_texts, "Dataset1", token2vec, tokenizer_encoder,
        model_max_seq_len, pad_idx, uniform_padding_length, selected_device, inference_batch_size
    )
    if embeddings_d1.size == 0:
        raise ValueError("Failed to generate embeddings for Dataset 1.")

    embeddings_d2, filtered_texts_d2 = _get_embeddings_for_texts_core(
        dataset2_texts, "Dataset2", token2vec, tokenizer_encoder,
        model_max_seq_len, pad_idx, uniform_padding_length, selected_device, inference_batch_size
    )
    if embeddings_d2.size == 0:
        raise ValueError("Failed to generate embeddings for Dataset 2.")
    
    print(f"Generated {embeddings_d1.shape[0]} embeddings for Dataset 1.")
    print(f"Generated {embeddings_d2.shape[0]} embeddings for Dataset 2.")

    # --- 5. Calculate Average Embeddings ---
    print("--- Calculating Average Embeddings ---")
    avg_embedding_d1 = np.mean(embeddings_d1, axis=0)
    avg_embedding_d2 = np.mean(embeddings_d2, axis=0)

    # --- NEW: Calculate Intra-Dataset Spread ---
    # Avg distance of individual embeddings to their dataset's mean
    avg_spread_d1 = np.mean(np.linalg.norm(embeddings_d1 - avg_embedding_d1, axis=1))
    avg_spread_d2 = np.mean(np.linalg.norm(embeddings_d2 - avg_embedding_d2, axis=1))
    print(f"Average intra-dataset spread for Dataset 1: {avg_spread_d1:.4f}")
    print(f"Average intra-dataset spread for Dataset 2: {avg_spread_d2:.4f}")

    # --- 6. Compute Similarities and Distances ---
    print("--- Computing Similarities and Distances ---")
    # Cosine Similarity between average embeddings
    similarity_avgD1_vs_avgD2 = cosine_similarity(
        avg_embedding_d1.reshape(1, -1), avg_embedding_d2.reshape(1, -1)
    )[0, 0]
    print(f"Cosine Similarity [Avg(Dataset1) vs Avg(Dataset2)]: {similarity_avgD1_vs_avgD2:.4f}")

    # NEW: Euclidean Distance between average embeddings
    euclidean_dist_avgD1_vs_avgD2 = np.linalg.norm(avg_embedding_d1 - avg_embedding_d2)
    print(f"Euclidean Distance [Avg(Dataset1) vs Avg(Dataset2)]: {euclidean_dist_avgD1_vs_avgD2:.4f}")


    # --- 7. Prepare Results ---
    results_dict = {
        "similarity_avg_dataset1_vs_avg_dataset2": float(similarity_avgD1_vs_avgD2),
        "euclidean_dist_avg_dataset1_vs_avg_dataset2": float(euclidean_dist_avgD1_vs_avgD2), # New
        "avg_spread_dataset1": float(avg_spread_d1), # New
        "avg_spread_dataset2": float(avg_spread_d2), # New
        "num_embeddings_dataset1": embeddings_d1.shape[0],
        "num_embeddings_dataset2": embeddings_d2.shape[0],
        "uniform_padding_length_used": uniform_padding_length,
    }

    if return_individual_dataset2_similarities:
        similarities_individual_d2_vs_avgD1 = cosine_similarity(
            embeddings_d2, avg_embedding_d1.reshape(1, -1)
        )
        individual_results = [
            {"text_dataset2": filtered_texts_d2[i], "similarity_to_avg_dataset1": float(similarities_individual_d2_vs_avgD1[i, 0])}
            for i in range(len(filtered_texts_d2)) # Use length of filtered_texts_d2
        ]
        results_dict["individual_dataset2_similarities_vs_avg_dataset1"] = individual_results
        
        if similarities_individual_d2_vs_avgD1.size > 0:
            avg_of_individual_d2_sims = np.mean(similarities_individual_d2_vs_avgD1)
            results_dict["avg_similarity_of_individual_dataset2_vs_avg_dataset1"] = float(avg_of_individual_d2_sims)
            # This print was already in your thought process, so it's good
            print(f"Average of Individual Similarities [Each Dataset2 vs Avg(Dataset1)]: {avg_of_individual_d2_sims:.4f}")


    if return_average_embeddings_data:
        results_dict["avg_embedding_dataset1_data"] = avg_embedding_d1.tolist()
        results_dict["avg_embedding_dataset2_data"] = avg_embedding_d2.tolist()

    print("--- Analysis Complete ---")
    return results_dict