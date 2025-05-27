# %%
import torch
import random
import itertools # To help generate combinations
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- SONAR Tokenizer Loading ---
try:
    from sonar.models.sonar_text import load_sonar_tokenizer
    # Updated load_tokenizer function to provide access to encode_as_tokens
    def load_tokenizer(repo="text_sonar_basic_encoder"):
        """Loads the SONAR tokenizer and provides access to encoder methods."""
        print(f"Loading SONAR tokenizer from repo: {repo}")
        # Load the base tokenizer which contains methods like create_encoder and vocab_info
        orig_tokenizer = load_sonar_tokenizer(repo)
        # Create the specific encoder instance
        encoder = orig_tokenizer.create_encoder()
        vocab_size = orig_tokenizer.vocab_info.size
        print(f"Tokenizer encoder loaded. Vocab size: {vocab_size}")

        # Wrapper class to hold the encoder
        class TokenizerEncoderWrapper:
            def __init__(self, encoder):
                self._encoder = encoder
                # Store vocab info if needed, accessible via encoder.vocab_info typically
                self.vocab_info = getattr(encoder, 'vocab_info', None) # Or get from orig_tokenizer

            def encode(self, text):
                """Encodes text into token IDs."""
                return self._encoder(text)

            def encode_as_tokens(self, text):
                """Encodes text into token strings using the encoder."""
                # Call encode_as_tokens directly on the encoder object
                if hasattr(self._encoder, 'encode_as_tokens'):
                    return self._encoder.encode_as_tokens(text)
                else:
                    # Fallback or warning if not found on the encoder
                    print(f"Warning: encode_as_tokens not found on the encoder object ({type(self._encoder)}).")
                    return None # Indicate failure

            # Optional: Allow calling the wrapper like the encoder
            def __call__(self, text):
                return self.encode(text)

        # Pass the encoder instance to the wrapper
        return TokenizerEncoderWrapper(encoder)

except ImportError:
    print("ERROR: SONAR library not found or load_sonar_tokenizer failed.")
    print("Using a dummy tokenizer.")
    class DummyTokenizerEncoder:
        vocab_size = 10
        def encode(self, text): return torch.tensor([random.randint(0,9) for _ in text.split()])
        def encode_as_tokens(self, text): return [f"tok_{i}" for i in range(len(text.split()))]
        def __call__(self, text): return self.encode(text)
    tokenizer_encoder = DummyTokenizerEncoder()
except Exception as e:
    print(f"An unexpected error occurred during tokenizer loading: {e}")
    raise

# %%
if __name__ == "__main__":
    # Load the tokenizer wrapper
    try:
        tokenizer_encoder = load_tokenizer()
    except Exception as e:
        print(f"Failed to load tokenizer, exiting. Error: {e}")
        exit()

    # Define word lists for sentence generation
    subjects_singular = ["He", "She", "It", "Cat", "Dog", "Man", "Woman"]
    subjects_plural = ["They", "We"]
    adverbs = ["", " very", " slightly"]
    predicates = ["hungry", "happy", "tired", "here", "home", "black", "white", "tall", "kind"]

    print("\n--- Generating and Tokenizing Sentences ---")

    # Generate sentences with "is"
    for subj, adverb, pred in itertools.product(subjects_singular, adverbs, predicates):
        sentence = f"{subj} is{adverb} {pred}"
        print(f"Sentence: \"{sentence}\"")
        try:
            token_ids = tokenizer_encoder.encode(sentence)
            print(f"  Token IDs: {token_ids.tolist()}")

            token_strings = tokenizer_encoder.encode_as_tokens(sentence)
            if token_strings is not None:
                print(f"  Tokens   : {token_strings}")
            else:
                print("  Tokens   : (encode_as_tokens not available or failed)")

        except Exception as e:
            print(f"  Error tokenizing sentence: {e}")
        finally:
            print("-" * 20)

    # Generate sentences with "are"
    for subj, pred in itertools.product(subjects_plural, predicates):
        sentence = f"{subj} are {pred}"
        print(f"Sentence: \"{sentence}\"")
        try:
            token_ids = tokenizer_encoder.encode(sentence)
            print(f"  Token IDs: {token_ids.tolist()}")

            token_strings = tokenizer_encoder.encode_as_tokens(sentence)
            if token_strings is not None:
                print(f"  Tokens   : {token_strings}")
            else:
                print("  Tokens   : (encode_as_tokens not available or failed)")

        except Exception as e:
            print(f"  Error tokenizing sentence: {e}")
        finally:
            print("-" * 20)


# %% Load Sonar encoder and generate embeddings for tokenized sentences
import torch
import matplotlib.pyplot as plt
import numpy as np
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

print("\n--- Generating Embeddings for Sentences ---")

# Load the Sonar encoder model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# check if local context has text2vec already loaded
if not hasattr(locals(), 'text2vec'):
    text2vec = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device
    )

# Store sentences and their embeddings
all_sentences = []
all_embeddings = []

# Process singular sentences
for subj, adverb, pred in itertools.product(subjects_singular, adverbs, predicates):
    sentence = f"{subj} is{adverb} {pred}"
    all_sentences.append(sentence)

    # Get embedding
    with torch.no_grad():
        embedding = text2vec.predict([sentence], "eng_Latn")[0]
    all_embeddings.append(embedding.cpu().numpy())

    print(f"Generated embedding for: \"{sentence}\"")

# Process plural sentences
for subj, adverb, pred in itertools.product(subjects_plural, adverbs, predicates):
    sentence = f"{subj} are{adverb} {pred}"
    all_sentences.append(sentence)

    # Get embedding
    with torch.no_grad():
        embedding = text2vec.predict([sentence], "eng_Latn")[0]
    all_embeddings.append(embedding.cpu().numpy())

    print(f"Generated embedding for: \"{sentence}\"")

# Convert to numpy array for analysis
embeddings_array = np.array(all_embeddings)

# %%
# Visualize embeddings using PCA
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
# Reduce to 2 dimensions for visualization using PHATE
from phate import PHATE
phate_operator = PHATE(n_components=2, random_state=42)
embeddings_2d = phate_operator.fit_transform(embeddings_array)

# Create a DataFrame for plotly
df = pd.DataFrame({
    'PHATE1': embeddings_2d[:, 0],
    'PHATE2': embeddings_2d[:, 1],
    'sentence': all_sentences
})

# Create interactive plot with plotly
fig = px.scatter(
    df, x='PHATE1', y='PHATE2',
    hover_data=['sentence'],
    text='sentence',
    opacity=0.7,
    title='PCA Visualization of Sentence Embeddings'
)

# Customize the appearance
fig.update_traces(
    textposition='top center',
    textfont=dict(size=8),
    marker=dict(size=8)
)

fig.update_layout(
    xaxis_title='PHATE 1',
    yaxis_title='PHATE 2',
    plot_bgcolor='rgba(240, 240, 240, 0.5)',
    width=1000,
    height=800
)

# Add grid
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')


output_dir = "./data/"
# Ensure the output directory exists
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)


# Save as HTML for interactivity
fig.write_html('./data/sentence_embeddings_pca.html')

# Save as image
fig.write_image('./data/sentence_embeddings_pca.png')

# Show the plot
#fig.show()

print("\nAnalysis complete. Visualization saved to '../data/sentence_embeddings_pca.png' and '../data/sentence_embeddings_pca.html'")
# %%


# %%
# Cross-covariance Analysis: Embeddings vs. Token Presence
import numpy as np # Ensure numpy is imported in this scope

print("\n--- Cross-covariance Analysis ---")

# 1. Tokenize all sentences again to get token strings
all_token_lists = []
valid_indices = []
print("Tokenizing sentences for cross-covariance analysis...")
for i, sentence in enumerate(all_sentences):
    try:
        # Ensure tokenizer_encoder is accessible, assuming it was loaded successfully earlier
        token_strings = tokenizer_encoder.encode_as_tokens(sentence)
        if token_strings is not None:
            all_token_lists.append(token_strings)
            valid_indices.append(i)
        else:
            print(f"Warning: Failed to get tokens for sentence {i}: \"{sentence}\"")
    except NameError:
        print("Error: tokenizer_encoder not found. Was the tokenizer loaded successfully?")
        break # Stop if tokenizer isn't available
    except Exception as e:
        print(f"Warning: Error tokenizing sentence {i}: \"{sentence}\". Error: {e}")

if not valid_indices:
    print("Error: No sentences could be tokenized successfully. Skipping cross-covariance analysis.")
else:
    # Filter embeddings to match successfully tokenized sentences
    # Ensure embeddings_array is available from the previous cell
    try:
        valid_embeddings_array = embeddings_array[valid_indices]
        print(f"Successfully tokenized {len(valid_indices)} out of {len(all_sentences)} sentences.")
    except NameError:
        print("Error: embeddings_array not found. Were embeddings generated successfully?")
        valid_embeddings_array = None # Mark as unavailable

if valid_embeddings_array is not None and len(valid_indices) > 1:

    # 2. Identify Unique Tokens
    all_tokens_flat = [token for sublist in all_token_lists for token in sublist]
    unique_tokens = sorted(list(set(all_tokens_flat)))
    token_to_index = {token: idx for idx, token in enumerate(unique_tokens)}
    print(f"Found {len(unique_tokens)} unique tokens.")

    # 3. Create Token Presence Matrix (Y)
    num_valid_sentences = len(valid_indices)
    num_unique_tokens = len(unique_tokens)
    Y = np.zeros((num_valid_sentences, num_unique_tokens), dtype=np.float32) # Use float for centering

    for i, token_list in enumerate(all_token_lists):
        for token in token_list:
            if token in token_to_index: # Should always be true
                j = token_to_index[token]
                Y[i, j] = 1.0 # Mark presence

    # 4. Define Embeddings Matrix (X)
    X = valid_embeddings_array
    print(f"Shape of X (embeddings): {X.shape}")
    print(f"Shape of Y (token presence): {Y.shape}")


    # 5. Center the Data
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    # 6. Calculate Cross-Covariance Matrix
    # C_XY = X_centered^T Y_centered / (N-1)
    N = num_valid_sentences
    C_XY = (X_centered.T @ Y_centered) / (N - 1)
    print(f"Shape of C_XY (cross-covariance): {C_XY.shape}") # (embedding_dim, num_unique_tokens)

    # 7. Analyze Directions
    # Calculate the L2 norm (magnitude) of each column (covariance vector per token)
    token_covariance_norms = np.linalg.norm(C_XY, axis=0)

    # Create a list of (token, norm) pairs and sort by norm descending
    token_norm_pairs = sorted(zip(unique_tokens, token_covariance_norms), key=lambda item: item[1], reverse=True)

    print("\nTop 20 Tokens by Cross-Covariance Norm with Embeddings:")
    for i, (token, norm) in enumerate(token_norm_pairs[:20]):
        print(f"{i+1}. Token: '{token}', Norm: {norm:.4f}")

    # The columns of C_XY (e.g., C_XY[:, token_to_index['happy']]) represent directions
    # in the embedding space associated with the presence of that token.
    # Further analysis could involve projecting these directions onto the PHATE/PCA components.
    print("\nCross-covariance analysis finished.")

    # 8. Perform SVD on the Cross-Covariance Matrix
    print("\n--- SVD Analysis of Cross-Covariance Matrix (C_XY) ---")
    try:
        # Perform SVD: C_XY = U @ diag(s) @ Vh
        # U: (embedding_dim, k) - Directions in embedding space
        # s: (k,) - Singular values
        # Vh: (k, num_unique_tokens) - Directions in token space (transposed)
        # k = min(embedding_dim, num_unique_tokens)
        U, s, Vh = np.linalg.svd(C_XY, full_matrices=False) # Use full_matrices=False for efficiency

        print(f"SVD computed. Shapes: U={U.shape}, s={s.shape}, Vh={Vh.shape}")

        # Analyze Variance Explained by Singular Values
        squared_singular_values = s**2
        total_variance = np.sum(squared_singular_values)
        variance_explained_ratio = squared_singular_values / total_variance
        cumulative_variance_explained = np.cumsum(variance_explained_ratio)

        print("\nVariance Explained by Top Singular Values:")
        print("-" * 50)
        print("  # | Singular Value | Variance Explained | Cumulative Variance")
        print("-" * 50)
        for i in range(min(10, len(s))): # Print top 10 or fewer
            print(f"{i+1:3d} | {s[i]:14.4f} | {variance_explained_ratio[i]:18.4%} | {cumulative_variance_explained[i]:20.4%}")
        print("-" * 50)

        # Optional: Analyze top singular vectors
        # Top Left Singular Vector (U[:, 0]): Principal direction in embedding space
        # Top Right Singular Vector (Vh[0, :]): Principal pattern in token space
        print(f"\nTop direction in embedding space (U[:, 0] norm): {np.linalg.norm(U[:, 0]):.2f}") # Should be 1
        print(f"Top pattern in token space (Vh[0, :] norm)   : {np.linalg.norm(Vh[0, :]):.2f}") # Should be 1

        # You could further analyze Vh[0, :] by finding tokens with the largest absolute values
        # top_token_indices = np.argsort(np.abs(Vh[0, :]))[::-1]
        # print("\nTokens associated with the primary cross-covariance pattern (Vh[0, :]):")
        # for idx in top_token_indices[:10]:
        #    print(f"  Token: '{unique_tokens[idx]}', Weight: {Vh[0, idx]:.3f}")

    except np.linalg.LinAlgError as e:
        print(f"SVD computation failed: {e}")
    except Exception as e:
        print(f"An error occurred during SVD analysis: {e}")


elif len(valid_indices) <= 1:
     print("Need more than one valid sentence with successful tokenization to calculate cross-covariance.")
# %%


# %%
# Analyze Semantic Vector Difference: "white" vs "black"
import numpy as np # Ensure numpy is imported

print("\n--- Semantic Vector Difference Analysis ('white' vs 'black') ---")

# Ensure embeddings_array and all_sentences are available
if 'embeddings_array' not in locals() or 'all_sentences' not in locals():
    print("Error: embeddings_array or all_sentences not found. Please ensure previous cells ran successfully.")
else:
    # Create a mapping from sentence to index for quick lookup
    sentence_to_index = {sentence: i for i, sentence in enumerate(all_sentences)}

    diff_vectors = []
    pairs_indices = [] # Store tuples of (index_white, index_black)
    pairs_sentences = [] # Store tuples of (sentence_white, sentence_black)

    # Find pairs differing by "white" / "black" for "is" sentences
    white = "tall"
    black = "hungry"
    for subj, adverb in itertools.product(subjects_singular, adverbs):
        sent_white = f"{subj} is{adverb} {white}"
        sent_black = f"{subj} is{adverb} {black}"

        if sent_white in sentence_to_index and sent_black in sentence_to_index:
            idx_white = sentence_to_index[sent_white]
            idx_black = sentence_to_index[sent_black]

            emb_white = embeddings_array[idx_white]
            emb_black = embeddings_array[idx_black]

            diff_vectors.append(emb_black - emb_white)
            pairs_indices.append((idx_white, idx_black))
            pairs_sentences.append((sent_white, sent_black))
            # print(f"Found pair: '{sent_white}' / '{sent_black}'") # Optional: verbosity

    # Find pairs differing by "white" / "black" for "are" sentences
    for subj, adverb in itertools.product(subjects_plural, adverbs):
        # Note: The loop generating 'are' sentences used the adverb in the f-string
        sent_white = f"{subj} are{adverb} {white}"
        sent_black = f"{subj} are{adverb} {black}"

        if sent_white in sentence_to_index and sent_black in sentence_to_index:
            idx_white = sentence_to_index[sent_white]
            idx_black = sentence_to_index[sent_black]

            emb_white = embeddings_array[idx_white]
            emb_black = embeddings_array[idx_black]

            diff_vectors.append(emb_black - emb_white)
            pairs_indices.append((idx_white, idx_black))
            pairs_sentences.append((sent_white, sent_black))
            # print(f"Found pair: '{sent_white}' / '{sent_black}'") # Optional: verbosity

    if not diff_vectors:
        print("Could not find any pairs of sentences differing only by '{white}'/'{black}'.")
    else:
        # Calculate the average difference vector
        avg_diff_vector = np.mean(np.array(diff_vectors), axis=0)
        print(f"Found {len(pairs_indices)} pairs. Calculated average '{black}' - '{white}' difference vector.")

        all_mse_pred = []
        all_mse_base = []

        # Calculate MSEs for each pair
        for i, (idx_white, idx_black) in enumerate(pairs_indices):
            emb_white = embeddings_array[idx_white]
            emb_black = embeddings_array[idx_black]

            # Predict 'black' embedding using the average difference
            pred_black = emb_white + avg_diff_vector

            # Calculate MSEs
            mse_pred = np.mean((pred_black - emb_black)**2)
            mse_base = np.mean((emb_white - emb_black)**2)

            all_mse_pred.append(mse_pred)
            all_mse_base.append(mse_base)

            # Optional: Print MSE per pair
            # sent_white, sent_black = pairs_sentences[i]
            # print(f"Pair: '{sent_white}'/'{sent_black}' -> Base MSE: {mse_base:.6f}, Pred MSE: {mse_pred:.6f}")

        # Calculate and report average MSEs
        avg_mse_base = np.mean(all_mse_base)
        avg_mse_pred = np.mean(all_mse_pred)
        mse_reduction = avg_mse_base - avg_mse_pred
        mse_reduction_ratio = (mse_reduction / avg_mse_base) if avg_mse_base > 1e-9 else 0 # Avoid division by zero

        print(f"\nAverage Baseline MSE (emb_white vs emb_black): {avg_mse_base:.6f}")
        print(f"Average Predicted MSE (emb_white + avg_diff vs emb_black): {avg_mse_pred:.6f}")
        print(f"MSE Reduction: {mse_reduction:.6f}")
        print(f"MSE Reduction Ratio: {mse_reduction_ratio:.2%}")

        if mse_reduction_ratio > 0:
            print(f"\nConclusion: The average difference vector explains a portion of the variance between '{white}' and '{black}' sentences.")
        else:
            print(f"\nConclusion: The average difference vector does not consistently explain the variance between '{white}' and '{black}' sentences in this dataset.")

# %%

# Analyze the difference vector using the SONAR decoder
print("\n--- Analyzing Difference Vector with SONAR Decoder ---")

if len(diff_vectors) > 0:
    # Get the average difference vector
    avg_diff_vector = np.mean(np.array(diff_vectors), axis=0)

    # Convert to tensor for decoder
    avg_diff_tensor = torch.tensor(avg_diff_vector, device=device)

    # Check if text decoder is already loaded in local context
    if not 'text_decoder' in locals():
        try:
            print("Loading SONAR text decoder model...")
            from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

            # Load the SONAR decoder model
            text_decoder = EmbeddingToTextModelPipeline(
                decoder="text_sonar_basic_decoder",
                tokenizer="text_sonar_basic_decoder",
                device=device
            )
            print("SONAR text decoder model loaded successfully")
        except Exception as e:
            print(f"Failed to load SONAR text decoder model: {e}")
            text_decoder = None

    # Select a few example pairs for demonstration
    num_examples = min(5, len(pairs_indices))

    if 'text_decoder' in locals() and text_decoder is not None:
        print(f"\nDecoding examples using the difference vector:")
        print(f"{'Original Sentence':<40} | {'Actual Modified':<40} | {'Predicted Modified':<40}")
        print("-" * 125)
        # Prepare all embeddings at once for batch processing
        white_embeddings = []
        white_sentences = []
        black_sentences = []

        for i in range(num_examples):
            idx_white, idx_black = pairs_indices[i]
            sent_white, sent_black = pairs_sentences[i]

            # Get the original embedding
            emb_white = torch.tensor(embeddings_array[idx_white], device=device)

            # Create a modified embedding by adding the difference vector
            modified_emb = emb_white + avg_diff_tensor

            white_embeddings.append(modified_emb)
            white_sentences.append(sent_white)
            black_sentences.append(sent_black)

        # Stack all embeddings for batch processing
        stacked_embeddings = torch.stack(white_embeddings)

        try:
            # Decode all embeddings in parallel
            with torch.no_grad():
                predicted_texts = text_decoder.predict(
                    stacked_embeddings,
                    "eng_Latn"
                )

            # Print results
            for i in range(num_examples):
                print(f"{white_sentences[i]:<40} | {black_sentences[i]:<40} | {predicted_texts[i]:<40}")

        except Exception as e:
            print(f"Error decoding embeddings in batch: {e}")

        print("\nNote: The predicted sentences are approximations based on the SONAR decoder's ability to decode embeddings.")
    else:
        print("SONAR text decoder model not available. Please check if SONAR is properly installed.")
else:
    print("Cannot analyze with decoder: no difference vectors found.")



# %%
# Out-of-Distribution Analysis with Complex Sentences
print("\n--- Out-of-Distribution Analysis with Complex Sentences ---")

if 'text_decoder' in locals() and text_decoder is not None and 'text2vec' in locals():
    # Define complex out-of-distribution sentences containing "tall"
    ood_sentences = [
        "The tall skyscraper dominated the city's skyline for decades",
        "Several remarkably tall basketball players were recruited from international leagues",
        "Ancient redwood trees, known for being incredibly tall, can live for thousands of years",
        "The professor described a theoretical model with tall probability distributions",
        "Renaissance artists often depicted mythological figures as unnaturally tall and powerful"
    ]

    print(f"Analyzing {len(ood_sentences)} complex out-of-distribution sentences containing 'tall'...")

    # Get embeddings for OOD sentences
    ood_embeddings = []
    for sentence in ood_sentences:
        with torch.no_grad():
            embedding = text2vec.predict([sentence], "eng_Latn")[0]
        ood_embeddings.append(embedding.cpu())

    # Stack embeddings for batch processing
    stacked_ood_embeddings = torch.stack(ood_embeddings).to(device)

    # Apply the difference vector to each embedding
    if 'avg_diff_tensor' in locals() and avg_diff_tensor is not None:
        modified_ood_embeddings = stacked_ood_embeddings + avg_diff_tensor

        try:
            # Decode original and modified embeddings
            with torch.no_grad():
                original_decoded = text_decoder.predict(
                    stacked_ood_embeddings,
                    "eng_Latn"
                )

                modified_decoded = text_decoder.predict(
                    modified_ood_embeddings,
                    "eng_Latn"
                )

            # Print results
            print(f"\nOut-of-Distribution Results (tall → hungry conversion):")
            print(f"{'Original Sentence':<60} | {'Original Decoded':<60} | {'Modified Decoded':<60}")
            print("-" * 185)

            for i, sentence in enumerate(ood_sentences):
                print(f"{sentence:<60} | {original_decoded[i]:<60} | {modified_decoded[i]:<60}")

            # Check if "hungry" appears in the modified sentences
            hungry_count = sum(1 for text in modified_decoded if "hungry" in text.lower())
            tall_count = sum(1 for text in modified_decoded if "tall" in text.lower())

            print(f"\nAnalysis: {hungry_count}/{len(ood_sentences)} modified sentences contain 'hungry'")
            print(f"Analysis: {tall_count}/{len(ood_sentences)} modified sentences still contain 'tall'")

        except Exception as e:
            print(f"Error decoding OOD embeddings: {e}")
    else:
        print("Cannot perform OOD analysis: no difference vector available.")
else:
    print("Cannot perform OOD analysis: required models not available.")


# %%

