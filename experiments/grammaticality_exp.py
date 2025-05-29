# %%
from sklearn.metrics import classification_report, confusion_matrix
import torch
import random
import sys
import os
import torch
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import numpy as np
import os
import torch
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm
import umap
import phate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.pca_utils import (
    perform_dimensionality_reduction,
    create_plot_dataframe
)
from sonar.models.sonar_text import (
    load_sonar_tokenizer,
)
from src.utils.generate_random_sequences import generate_random_sequences

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

def generate_pca_plots_from_datasets(
    datasets: list,
    labels: list,
    datasets_paths: list = None,
    output_dir: str = OUTPUT_DIR,
    n_components: int = 3,
    reduction_method: str = "PCA", # Added parameter for dimensionality reduction method
    enable_correctness_direction_analysis: bool = True, # Enable or disable grammaticality direction analysis
    return_eigenvectors: bool = False,
):
    """
    Generates dimensionality reduction plots (PCA, PHATE, or UMAP) for embeddings from multiple datasets.

    Args:
        datasets (list): List of paths to the dataset files (e.g., ["legal_code.txt", "illegal_code.txt"]).
        labels (list): List of labels corresponding to each dataset (e.g., ['legal', 'illegal']).
        output_dir (str): Directory for output files (not explicitly used for saving plots in this version).
        n_components (int): Number of components for dimensionality reduction (2 or 3).
        reduction_method (str): Method for dimensionality reduction ('PCA', 'PHATE', or 'UMAP').
    """
    if labels is None:
        raise ValueError("Labels must be provided.")
    if datasets is None and datasets_paths is None:
        raise ValueError("Either 'datasets' (list of texts) or 'datasets_paths' (list of file paths) must be provided.")
    if datasets is not None and datasets_paths is not None:
        raise ValueError("Provide either 'datasets' or 'datasets_paths', not both.")
    if datasets is not None and len(datasets) != len(labels):
        raise ValueError("Length of 'datasets' and 'labels' must match.")
    if datasets_paths is not None and len(datasets_paths) != len(labels):
        raise ValueError("Length of 'datasets_paths' and 'labels' must match.")

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
    if datasets_paths is not None:
        for i in range(len(datasets_paths)):
            dataset_path = datasets_paths[i]
            label = labels[i]
            
            with open(dataset_path, "r") as f:
                texts = [line for line in f.read().split("\n") if line.strip()] # Remove empty lines

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
    elif datasets is not None:
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
    
    print(f"Maximum sequence length observed across all datasets: {max_len_overall}")

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
    print(f"Shape of combined raw embeddings for reduction: {combined_raw_embeddings.shape}")

    # --- Perform Dimensionality Reduction ---
    print(f"--- Performing {reduction_method} with {n_components} components ---")
    reduced_embeddings = None
    if reduction_method == "PCA":
        operator = PCA(n_components=n_components, random_state=RANDOM_STATE)
        reduced_embeddings = operator.fit_transform(combined_raw_embeddings)
    elif reduction_method == "UMAP":
        operator = umap.UMAP(n_components=n_components, random_state=RANDOM_STATE)
        reduced_embeddings = operator.fit_transform(combined_raw_embeddings)
    elif reduction_method == "PHATE":
        operator = phate.PHATE(n_components=n_components, random_state=RANDOM_STATE)
        reduced_embeddings = operator.fit_transform(combined_raw_embeddings)
    else:
        raise ValueError(f"Unknown reduction_method: {reduction_method}. Choose from 'PCA', 'UMAP', 'PHATE'.")

    # --- Create Plot Dataframe ---
    df_dict = {
        f'{reduction_method}1': reduced_embeddings[:, 0],
        f'{reduction_method}2': reduced_embeddings[:, 1],
        'label': labels_vec,
        'sentence': combined_raw_texts # Use 'sentence' for hover text
    }
    if n_components == 3:
        df_dict[f'{reduction_method}3'] = reduced_embeddings[:, 2]

    df = pd.DataFrame(df_dict)
    
    title = f'{reduction_method} of {" vs ".join(labels)} Embeddings ({n_components}D)'
    
    print("--- Generating Plots ---")
    if n_components == 2:
        fig_interactive = px.scatter(
            df, x=f'{reduction_method}1', y=f'{reduction_method}2', color='label',
            title=title,
            hover_data={'sentence': True, f'{reduction_method}1':':.3f', f'{reduction_method}2':':.3f'},
            width=900, height=700,
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        # Enhanced styling
        fig_interactive.update_traces(
            marker=dict(size=8, opacity=0.7, line=dict(width=1, color='white'))
        )
        fig_interactive.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='gray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='gray'),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01)
        )
        
    elif n_components == 3:
        fig_interactive = px.scatter_3d(
            df, x=f'{reduction_method}1', y=f'{reduction_method}2', z=f'{reduction_method}3', color='label',
            title=title,
            hover_data={'sentence': True, f'{reduction_method}1':':.3f', f'{reduction_method}2':':.3f', f'{reduction_method}3':':.3f'},
            width=1000, height=800,
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        # Enhanced styling
        fig_interactive.update_traces(
            marker=dict(size=6, opacity=0.8, line=dict(width=0.5, color='white'))
        )
        fig_interactive.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            scene=dict(
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                xaxis=dict(title=f'{reduction_method} Component 1'),
                yaxis=dict(title=f'{reduction_method} Component 2'),
                zaxis=dict(title=f'{reduction_method} Component 3')
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01)
        )
    else:
        raise ValueError("n_components must be 2 or 3 for plotting.")

    fig_interactive.show()

    # --- Calculate separation metrics (for 2 components as silhouette_score works best) ---
    if n_components >= 2:
        silhouette_avg = silhouette_score(reduced_embeddings[:, :2], labels_vec)
        print(f"Silhouette Score (using first 2 components): {silhouette_avg}")
    else:
        print("Silhouette Score requires at least 2 components for calculation.")

    # Initialize variable for return, to be defined if analysis runs
    correctness_direction_to_return = None 

    # --- Binary Label Direction Analysis ---
    if enable_correctness_direction_analysis:
        binary_labels = []
        has_binary_labels = False

        label_set = set(labels_vec)  # Use set to avoid duplicates
        if len(label_set) < 2:
            print("\nSkipping binary direction analysis: Not enough unique labels found.")
            return reduced_embeddings, df, fig_interactive, None, None
        elif len(label_set) > 2:
            print("\nWarning: More than 2 unique labels found. Not implemented yet")
            print("Binary direction analysis is designed for exactly 2 labels.")
        else:
            # Get the two unique labels
            unique_labels = list(label_set)
            label_1, label_2 = unique_labels[0], unique_labels[1]
            print(f"\nFound exactly 2 unique labels: '{label_1}' and '{label_2}', proceeding with binary direction analysis.")
            
            # Convert labels to binary (0/1)
            binary_labels = []
            for label in labels_vec:
                if label == label_1:
                    binary_labels.append(1)
                else:
                    binary_labels.append(0)
            
            has_binary_labels = True

        if has_binary_labels and len(set(binary_labels) - {-1}) == 2:
            print(f"\n--- Performing Binary Direction Analysis ({label_1} vs {label_2}) ---")

            # Filter out entries that are not in our binary classification
            valid_indices = [i for i, val in enumerate(binary_labels) if val != -1]
            X_filtered = combined_raw_embeddings[valid_indices]
            y_filtered = np.array([label for label in binary_labels if label != -1])

            if len(set(y_filtered)) < 2:
                print("Warning: Not enough unique classes found for direction analysis.")
            elif len(X_filtered) < 2:
                print("Warning: Not enough data points after filtering for binary direction analysis.")
            else:
                try:
                    # Train-test split to avoid overfitting
                    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
                        X_filtered, y_filtered, valid_indices, test_size=0.2, random_state=RANDOM_STATE, 
                        stratify=y_filtered if len(np.unique(y_filtered)) > 1 else None
                    )
                    
                    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
                    
                    # Using Logistic Regression to find the separating hyperplane/direction
                    log_reg_model = LogisticRegression(random_state=RANDOM_STATE, solver='liblinear', C=1.0, max_iter=1000)
                    log_reg_model.fit(X_train, y_train)

                    # Calculate accuracy on both training and test sets
                    train_accuracy = log_reg_model.score(X_train, y_train)
                    test_accuracy = log_reg_model.score(X_test, y_test)
                    print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
                    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
                    
                    if train_accuracy - test_accuracy > 0.1:
                        print("⚠️  Warning: Large gap between training and test accuracy suggests overfitting")

                    # Get predictions for test set
                    y_test_pred = log_reg_model.predict(X_test)

                    # Optional: Add more detailed metrics (on test set)
                    print(f"\nTest Set Classification Report:")
                    print(classification_report(y_test, y_test_pred, target_names=[label_2, label_1]))

                    print(f"\nTest Set Confusion Matrix:")
                    cm = confusion_matrix(y_test, y_test_pred)
                    print(cm)
                    print("(rows: actual, columns: predicted)")

                    direction_unnormalized = log_reg_model.coef_[0]
                    direction_normalized = direction_unnormalized / np.linalg.norm(direction_unnormalized)
                    direction_to_return = direction_normalized

                    print(f"\nLearned Normalized Direction Vector (Logistic Regression coefficients):\n{direction_normalized[:10]}... (showing first 10 dims)")
                    print(f"This vector is in the original embedding space. A higher dot product with this vector generally means more '{label_1}'-like.")

                    # Calculate direction scores for ALL filtered data (not just test set)
                    direction_scores = np.dot(X_filtered, direction_normalized)

                    print(f"\nExample Direction Scores (from test set):")
                    # Show examples from test set
                    test_scores = np.dot(X_test, direction_normalized)
                    for i in range(min(5, len(test_scores))):
                        text_index = test_indices[i]
                        predicted_label = label_1 if y_test_pred[i] == 1 else label_2
                        actual_label = label_1 if y_test[i] == 1 else label_2
                        correct = "✓" if y_test_pred[i] == y_test[i] else "✗"
                        
                        if text_index < len(combined_raw_texts):
                            print(f"  Sentence: '{combined_raw_texts[text_index][:50]}...' -> Score: {test_scores[i]:.4f}")
                            print(f"    Actual: {actual_label}, Predicted: {predicted_label} {correct}")
                        else:
                            print(f"  Score: {test_scores[i]:.4f} (Actual: {actual_label}, Predicted: {predicted_label} {correct}) (Text index out of bounds)")

                    # --- Histogram of Direction Scores ---
                    print(f"\n--- Direction Scores Distribution ({label_1} vs {label_2}) ---")
                    import matplotlib.pyplot as plt
                    
                    label_1_scores = direction_scores[y_filtered == 1]
                    label_2_scores = direction_scores[y_filtered == 0]
                    
                    plt.figure(figsize=(10, 6))
                    plt.hist(label_2_scores, bins=20, alpha=0.7, label=label_2, color='red', density=True)
                    plt.hist(label_1_scores, bins=20, alpha=0.7, label=label_1, color='green', density=True)
                    plt.xlabel('Direction Score')
                    plt.ylabel('Density')
                    plt.title(f'Distribution of Direction Scores ({label_1} vs {label_2})')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Save histogram
                    hist_filename = f"direction_scores_histogram_{labels[0].split('_')[1]}_{n_components}D_{reduction_method}.png"
                    hist_filepath = os.path.join(output_dir, hist_filename)
                    plt.savefig(hist_filepath, dpi=300, bbox_inches='tight')
                    print(f"Direction scores histogram saved to: {hist_filepath}")
                    plt.close()
                    
                    # Print some statistics about the distributions
                    print(f"\n{label_1} scores - Mean: {np.mean(label_1_scores):.4f}, Std: {np.std(label_1_scores):.4f}")
                    print(f"{label_2} scores - Mean: {np.mean(label_2_scores):.4f}, Std: {np.std(label_2_scores):.4f}")
                    print(f"Separation (difference in means): {np.mean(label_1_scores) - np.mean(label_2_scores):.4f}")

                    # --- Sanity Check: Average Direction Scores ---
                    print(f"\n--- Sanity Check: Average Direction Scores ---")
                    avg_label_1_score = None  # Initialize
                    if len(label_1_scores) > 0:
                        avg_label_1_score = np.mean(label_1_scores)
                        print(f"Average score for '{label_1}' sentences: {avg_label_1_score:.4f}")
                    else:
                        print(f"No '{label_1}' sentences found for average score calculation.")

                    avg_label_2_score = None  # Initialize
                    if len(label_2_scores) > 0:
                        avg_label_2_score = np.mean(label_2_scores)
                        print(f"Average score for '{label_2}' sentences: {avg_label_2_score:.4f}")
                    else:
                        print(f"No '{label_2}' sentences found for average score calculation.")

                    if avg_label_1_score is not None and avg_label_2_score is not None:
                        if avg_label_2_score < avg_label_1_score:
                            print(f"Observation: Average score for '{label_2}' sentences is lower than for '{label_1}' ones, as expected.")
                        else:
                            print(f"Observation: Average score for '{label_2}' sentences is NOT lower than '{label_1}'. This may indicate issues or interesting properties.")
                    

                    # --- Sanity Check: Cosine Similarity (PCA PC1 vs Direction Vector) ---
                    if return_eigenvectors and reduction_method == "PCA" and hasattr(operator, 'components_') and operator.components_ is not None:
                        pc1 = operator.components_[0]  # PC1 from PCA
                        if pc1.shape == direction_normalized.shape and pc1.ndim == 1:
                            # PC1 from sklearn.decomposition.PCA is already a unit vector.
                            # direction_normalized is also a unit vector.
                            cosine_sim = np.dot(pc1, direction_normalized)
                            print(f"\n--- Sanity Check: Cosine Similarity (PC1 vs Normalized Direction Vector) ---")
                            print(f"Cosine similarity: {cosine_sim:.4f}")
                            print("Interpretation: A high absolute value (close to 1 or -1) suggests PC1 (direction of max variance)")
                            print(f"aligns with the learned direction vector ({label_1} vs {label_2}). A value close to 0 suggests they are orthogonal.")
                            print(f"This alignment is plausible if the '{label_1}' vs '{label_2}' distinction is a primary driver of variance captured by PC1.")
                        else:
                            print("\nSkipping cosine similarity check: PC1 or direction vector has unexpected shape or dimensions.")
                            print(f"PC1 shape: {pc1.shape}, Norm Direction Vec shape: {direction_normalized.shape}")
                    elif reduction_method != "PCA":
                        print("\nCosine similarity check with eigenvectors is specific to PCA; current method is not PCA.")
                    elif not return_eigenvectors:
                        print("\nSkipping cosine similarity check: Eigenvectors (PCA components) were not requested to be returned.")
                    else:  # operator.components_ is None or not available
                        print("\nSkipping cosine similarity check: PCA components not available.")

                except Exception as e:
                    print(f"Error during binary direction analysis: {e}")
                    print("Ensure sufficient data points for both label classes.")
        else:
            print(f"\nBinary direction analysis skipped: Labels do not contain exactly 2 categories, or only one category was found after filtering, or not enough data.")

    # Prepare components for return (specific to PCA)
    pca_components_to_return = None
    if return_eigenvectors and reduction_method == "PCA" and hasattr(operator, 'components_'):
        pca_components_to_return = operator.components_

    # --- Saving results before returning ---
    # Make sure 'os' module is imported at the top of your script (e.g., import os)
    # Make sure 'numpy' is imported (e.g., import numpy as np)
    # pandas and plotly.express are assumed to be imported as they are used earlier.

    print(f"\n--- Saving Results to {output_dir} ---")
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Save reduced embeddings (using filename structure from your example)
    reduced_filename = f"reduced_embeddings_{labels[0].split('_')[1]}_{n_components}D_{reduction_method}.npy"
    np.save(os.path.join(output_dir, reduced_filename), reduced_embeddings)
    print(f'Reduced embeddings saved to: {os.path.join(output_dir, reduced_filename)}')

    # Save DataFrame (using filename structure from your example)
    df_filename = f"plot_dataframe_{labels[0].split('_')[1]}_{n_components}_{reduction_method}.csv"
    df.to_csv(os.path.join(output_dir, df_filename), index=False)
    print(f"Plot dataframe saved to: {os.path.join(output_dir, df_filename)}")

    # Create a descriptive filename for the interactive plot (as per your request)
    # The 'title' variable is: f'{reduction_method} of {" vs ".join(labels)} Embeddings ({n_components}D)'

    # Sanitize the title string to create a valid filename component
    # First, replace common separators and problematic characters with underscores
    plot_filename_base_intermediate = title.replace(' ', '_').replace(':', '_').replace('/', '_').replace('\\', '_')
    plot_filename_base_intermediate = plot_filename_base_intermediate.replace('(', '_').replace(')', '_').replace('[', '_').replace(']', '_')

    # Define a set of allowed characters for filenames (alphanumeric, underscore, hyphen)
    # Dot is generally for extension, so avoid in base name unless intended.
    allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-')
    sanitized_plot_filename_base = "".join(c if c in allowed_chars else '_' for c in plot_filename_base_intermediate)

    # Consolidate multiple underscores that might result from replacements and remove leading/trailing ones
    sanitized_plot_filename_base = '_'.join(filter(None, sanitized_plot_filename_base.split('_')))

    if not sanitized_plot_filename_base:  # Fallback if title becomes empty or invalid after sanitization
        sanitized_plot_filename_base = f"{reduction_method.lower()}_{'_'.join(labels)}_{n_components}D_interactive_plot"
    plot_html_filename = f"{sanitized_plot_filename_base}.html"

    fig_interactive.write_html(os.path.join(output_dir, plot_html_filename))
    print(f"Interactive plot saved to: {os.path.join(output_dir, plot_html_filename)}")

    plot_png_filename = "static_" + plot_html_filename.replace(".html", ".png")
    png_filepath = os.path.join(output_dir, plot_png_filename)

    try:
        fig_interactive.write_image(png_filepath)
        print(f"Static PNG plot saved to: {png_filepath}")
    except ValueError as e:
        print(f"Error saving PNG: {e}")
        print("Please ensure you have either 'kaleido' or 'orca' installed and in your PATH.")
        print("You can install kaleido with: pip install kaleido")
        print("Or for orca (more complex): https://plotly.com/python/static-image-export/")

    # Save PCA components (eigenvectors) if available (using filename from your example)
    # The variable in your function for eigenvectors is 'pca_components_to_return'
    if pca_components_to_return is not None:
        pca_filename = f"pca_eigenvectors_{labels[0].split('_')[1]}_{n_components}D_{reduction_method}.npy"
        np.save(os.path.join(output_dir, pca_filename), pca_components_to_return)
        print(f"PCA eigenvectors saved to: {os.path.join(output_dir, pca_filename)}")

    # Save direction vector if available (using filename from your example)
    # The variable in your function for direction vector is 'direction_to_return'
    if direction_to_return is not None:
        direction_filename = f"direction_vector_{labels[0].split('_')[1]}_{n_components}D_{reduction_method}.npy"
        np.save(os.path.join(output_dir, direction_filename), direction_to_return)
        print(f"Direction vector saved to: {os.path.join(output_dir, direction_filename)}")

    print(f"Results saved to output directory: {output_dir}")

    return reduced_embeddings, df, fig_interactive, pca_components_to_return, direction_to_return
#%%

# reduced_embeddings, df, fig_interactive, eigenvectors, grammaticality_direction = generate_pca_plots_from_datasets(datasets=["../data/simple_maths.txt", "../data/incorrect_simple_maths.txt"], labels=["grammatical_maths", "agrammar_maths"], n_components=2, reduction_method="PCA", return_eigenvectors=True, enable_grammaticality_direction_analysis=True)

# %%
