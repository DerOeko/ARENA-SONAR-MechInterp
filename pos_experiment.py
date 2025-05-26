#%%
import torch
import random
import itertools
import matplotlib.pyplot as plt 
import numpy as np
import os
from tqdm import tqdm
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import torch
from fairseq2.data import Collater
from fairseq2.data.data_pipeline import read_sequence
from fairseq2.data.text import TextTokenizer, read_text
from fairseq2.generation import (
    BeamSearchSeq2SeqGenerator,
    Sampler,
    SamplingSeq2SeqGenerator,
    Seq2SeqGenerator,
    SequenceToTextConverter,
    TextTranslator,
)
from fairseq2.typing import CPU, DataType, Device

from sonar.inference_pipelines.utils import add_progress_bar, extract_sequence_batch
from sonar.models.encoder_model import SonarEncoderModel
from sonar.models.sonar_text import (
    load_sonar_text_decoder_model,
    load_sonar_text_encoder_model,
    load_sonar_tokenizer,
)
from sonar.models.sonar_translation import SonarEncoderDecoderModel
from sonar.models.sonar_translation.model import DummyEncoderModel
from sonar.nn.conditional_decoder_model import ConditionalTransformerDecoderModel
# SONAR and fairseq2 imports
from sonar.models.sonar_text import load_sonar_tokenizer
from sonar.models.encoder_model import SonarEncoderModel # For type hinting
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, extract_sequence_batch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask
from fairseq2.data import Collater 
from fairseq2.typing import Device, DataType, CPU
from fairseq2.models.sequence import SequenceModelOutput
from typing import Optional, Tuple
# Plotting
import plotly.express as px
import pandas as pd
from phate import PHATE
global VOCAB_INFO, PAD_IDX, EOS_IDX, UNK_IDX, ENG_LANG_TOKEN_IDX, DEVICE, RANDOM_STATE, MODEL_NAME, OUTPUT_DIR

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
MODEL_NAME = "text_sonar_basic_encoder"

WORDS_TO_TEST = ["dog", "cat", "car", "house", "tree", "love", "run", "code", "data", "model"]

OUTPUT_DIR = "./data/"

# Seed for reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(RANDOM_STATE)

# %% --- Tokenizer and Special Token IDs ---
print("--- Initializing Tokenizer and Special IDs ---")
orig_sonar_tokenizer = load_sonar_tokenizer(MODEL_NAME)
tokenizer_encoder = orig_sonar_tokenizer.create_encoder()
tokenizer_decoder = orig_sonar_tokenizer.create_decoder()

VOCAB_INFO = orig_sonar_tokenizer.vocab_info
PAD_IDX = VOCAB_INFO.pad_idx
EOS_IDX = VOCAB_INFO.eos_idx
UNK_IDX = VOCAB_INFO.unk_idx
dummy_tokenized_for_special_tokens = tokenizer_encoder("test")
ENG_LANG_TOKEN_IDX = dummy_tokenized_for_special_tokens[0].item()

print(f"Using Language ID (eng_Latn): {ENG_LANG_TOKEN_IDX} ('{tokenizer_decoder(torch.tensor([ENG_LANG_TOKEN_IDX]))}')")
print(f"Using PAD ID: {PAD_IDX} ('{tokenizer_decoder(torch.tensor([PAD_IDX]))}')")

#%%
class CustomTextToEmbeddingPipeline(TextToEmbeddingModelPipeline):
    def __init__(self, encoder, 
                 tokenizer, # TextTokenizer from fairseq2.data.text
                 device: Device = CPU, 
                 dtype = None):
        super().__init__(encoder, tokenizer, device, dtype)
        # self.model is an instance of SonarTextTransformerEncoderModel

    @torch.inference_mode()
    def predict_from_token_ids(
        self,
        token_id_sequences: torch.Tensor, # Expects a 2D tensor [batch_size, seq_len]
        target_device: Optional[Device] = None,
        steering_vector: Optional[torch.Tensor] = None,
        target_token_indices_for_steering: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[PaddingMask]]: # Return type

        # Assertions for steering parameters
        if steering_vector is not None and target_token_indices_for_steering is None:
            raise ValueError("If steering_vector is provided, target_token_indices_for_steering must also be provided.")
        if steering_vector is None and target_token_indices_for_steering is not None:
            raise ValueError("If target_token_indices_for_steering is provided, steering_vector must also be provided.")

        # Input validation
        if not isinstance(token_id_sequences, torch.Tensor) or token_id_sequences.ndim > 2:
            raise ValueError("Input token_id_sequences must be a 2D torch.Tensor [batch_size, seq_len]")
        if token_id_sequences.ndim == 1:
            token_id_sequences = token_id_sequences.unsqueeze(0)

        token_ids_on_device = token_id_sequences.to(self.device)

        # 1. Create the initial PaddingMask based on input token IDs
        padding_mask_bool_tensor = (token_ids_on_device != self.tokenizer.vocab_info.pad_idx)
        seq_lengths = padding_mask_bool_tensor.sum(dim=1)
        # This is the padding_mask that corresponds to token_ids_on_device
        current_padding_mask = PaddingMask(seq_lengths, batch_seq_len=token_ids_on_device.shape[1])

        # --- Replicating SonarTextTransformerEncoderModel.forward() manually ---

        # 2. Encoder Frontend call
        # It returns: (embed_seqs_tensor, padding_mask_after_frontend)
        initial_embeds_tensor, pm_after_frontend = self.model.encoder_frontend(
            token_ids_on_device, current_padding_mask
        )

        # 3. Encoder call
        # It returns: (encoded_seqs_tensor, padding_mask_after_encoder)
        # It takes initial_embeds_tensor and its corresponding padding_mask (pm_after_frontend)
        encoded_seqs_tensor, pm_after_encoder = self.model.encoder(
            initial_embeds_tensor, pm_after_frontend
        )

        # This is the tensor for last_hidden_states before potential steering
        current_last_hidden_states = encoded_seqs_tensor 

        # 4. Optional LayerNorm (as in SonarTextTransformerEncoderModel.forward)
        if self.model.layer_norm is not None:
            current_last_hidden_states = self.model.layer_norm(current_last_hidden_states)
            # LayerNorm does not change the padding mask structure.
            # So, pm_after_encoder is still the valid padding mask for current_last_hidden_states.

        # Make a clone only if steering is applied and you need to preserve the pre-steered version
        # or if current_last_hidden_states comes from a part of graph that shouldn't be modified in-place.
        # For simplicity, let's assume we operate on this tensor directly or clone if steering.
        if steering_vector is not None:
            current_last_hidden_states = current_last_hidden_states.clone()


        # 5. Apply Steering (if parameters are provided)
        if steering_vector is not None and target_token_indices_for_steering is not None:
            steering_vector_dev = steering_vector.to(current_last_hidden_states.device)
            target_indices_dev = target_token_indices_for_steering.to(
                device=current_last_hidden_states.device, dtype=torch.long
            )

            if target_indices_dev.shape[0] != current_last_hidden_states.shape[0]:
                raise ValueError(f"Batch size of target_token_indices_for_steering ({target_indices_dev.shape[0]}) must match batch size of embeddings ({current_last_hidden_states.shape[0]}).")
            if not (steering_vector_dev.ndim == 1 and steering_vector_dev.shape[0] == current_last_hidden_states.shape[2]):
                 raise ValueError(f"Steering vector must be 1D and match embedding dimension ({current_last_hidden_states.shape[2]}). Got shape {steering_vector_dev.shape}.")

            batch_indices_for_steering = torch.arange(current_last_hidden_states.size(0), device=current_last_hidden_states.device)
            
            selected_tokens = current_last_hidden_states[batch_indices_for_steering, target_indices_dev]
            steered_tokens = selected_tokens + steering_vector_dev # Add steering vector
            current_last_hidden_states[batch_indices_for_steering, target_indices_dev] = steered_tokens
        
        # 6. Pool the (potentially layer_normed and steered) final hidden states
        # The `self.model.pool` method takes `seqs`, `padding_mask`, and the `pooling` strategy enum.
        # The `padding_mask` here should correspond to `current_last_hidden_states`.
        # In the original `SonarTextTransformerEncoderModel.forward`, it uses the `padding_mask` returned
        # from `self.encoder_frontend`. So, `pm_after_frontend` is the correct one to use for pooling.
        sentence_embeddings = self.model.pool(
            current_last_hidden_states, pm_after_frontend, self.model.pooling
        )
        
        # --- Results from manual path are now used ---
        final_target_device = target_device or self.device
        
        # The padding mask returned by the pipeline should be relevant to the *final* hidden states.
        # pm_after_encoder is the most accurate for current_last_hidden_states if no seq len changes.
        # However, SonarEncoderOutput returns padding_mask from frontend. For consistency with that,
        # we can return pm_after_frontend. They are usually the same.
        padding_mask_to_return = pm_after_frontend 
        
        return (
            sentence_embeddings.to(final_target_device),
            current_last_hidden_states.to(final_target_device), 
            initial_embeds_tensor.to(final_target_device),
            padding_mask_to_return # This is an Optional[PaddingMask] object
        )

#%%
text2vec = CustomTextToEmbeddingPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=DEVICE
)
MAX_SEQ_LEN = text2vec._modules['model'].encoder_frontend.pos_encoder.max_seq_len

# testing the pipeline
test_sentence = "The quick brown fox jumps over the lazy dog."

#test_encoded_list = tokenizer_encoder(test_sentence) 

#print(f"Tokenized sentence (tensor): {test_encoded_list}")

# Add batch dimension for predict_from_token_ids as it expects [batch_size, seq_len]
#test_embedding = text2vec.predict_from_token_ids(test_encoded_list) 

# Load Sonar encoder and generate embeddings for tokenized sentences
print("\n--- Generating Embeddings for Sentences ---")

words_to_test = ["house"]
word_token_ids = {word: tokenizer_encoder(word)[1] for word in words_to_test}
print(f"Token IDs for words {words_to_test}: {word_token_ids}")

all_token_sequences = []
all_labels = []
all_positions = []
for word_str in words_to_test:
    for i in range(1, MAX_SEQ_LEN - 1, 4):
        token_ids = torch.full((MAX_SEQ_LEN,), UNK_IDX, dtype=torch.long, device= DEVICE)
        token_ids[0] = 256047
        token_ids[-1] = EOS_IDX
        token_ids[i] = word_token_ids[word_str]
        
        all_token_sequences.append(token_ids)
        all_labels.append(f"{word_str}_pos{i}")
        all_positions.append(i)

# Convert to tensor
all_token_sequences = torch.stack(all_token_sequences).to(DEVICE)
all_positions = torch.tensor(all_positions, dtype=torch.long, device=DEVICE)#%% Get word embeddings
#%%
INFERENCE_BATCH_SIZE = 16
sentence_embeddings_list = [] # Storing sentence embeddings (previously word_embeddings)
last_token_embeddings_list = []    # Storing specific token hidden states (previously token_embeddings)
first_token_embeddings_list = [] # Storing first token hidden states (if needed)
with torch.no_grad():
    # Corrected loop iterator
    for batch_start_idx in tqdm(range(0, len(all_token_sequences), INFERENCE_BATCH_SIZE), desc="Generating Embeddings"):
        batch_end_idx = batch_start_idx + INFERENCE_BATCH_SIZE # Calculate end index for slicing
        
        batch_tokens_ids = all_token_sequences[batch_start_idx:batch_end_idx]
        batch_actual_positions = all_positions[batch_start_idx:batch_end_idx] # These are the 'i' values for target tokens

        if batch_tokens_ids.size(0) == 0: # Handle potential empty last batch if len % BATCH_SIZE == 0
            continue

        # Get sentence embeddings and all last hidden states for the batch
        batch_sentence_embeddings, batch_last_hidden_states, batch_first_hidden_states, _ = text2vec.predict_from_token_ids(batch_tokens_ids)
        
        # Store sentence embeddings (optional, if you still need them)
        sentence_embeddings_list.append(batch_sentence_embeddings.cpu().numpy())
        
        # Extract the hidden state of the target token using its actual position in the sequence
        # batch_indices will be [0, 1, ..., current_batch_size-1]
        batch_indices = torch.arange(batch_tokens_ids.size(0), device=DEVICE)
        
        # This is the key step:
        # batch_last_hidden_states has shape [current_batch_size, MAX_SEQ_LEN, hidden_dim]
        # batch_indices has shape [current_batch_size]
        # batch_actual_positions (your previous batch_word_ids) has shape [current_batch_size]
        # This correctly selects the hidden state vector for the target token at its specific position 'i'
        # for each sequence in the batch.
        specific_token_last_embeddings_batch = batch_last_hidden_states[batch_indices, batch_actual_positions]
        last_token_embeddings_list.append(specific_token_last_embeddings_batch.cpu().numpy())
        
        specific_token_initial_embeddings_batch = batch_first_hidden_states[batch_indices, batch_actual_positions]
        first_token_embeddings_list.append(specific_token_initial_embeddings_batch.cpu().numpy())
# Moved print statement outside the loop
print(f"Generated embeddings for words: {words_to_test}")

# Now, last_token_embeddings_list contains a list of numpy arrays,
# each array being the hidden states of your target token (e.g., "dog")
# from a batch. You'll concatenate them for PCA/PHATE.

# Example of concatenating for PCA:
if last_token_embeddings_list: # Check if the list is not empty
    embeddings_for_pca = np.concatenate(last_token_embeddings_list, axis=0)
    # Now 'embeddings_for_pca' is ready for your PCA and plotting code.
    # Ensure all_labels, words_for_df, positions_for_df align with this concatenated array.
else:
    print("No embeddings were generated.")
#%%
import torch # Needed for isinstance(all_positions, torch.Tensor) in create_dataframe
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import numpy as np
import os

# --- Helper Functions (from your code, slightly adjusted for clarity/consistency) ---
# Ensure RANDOM_STATE is defined globally if used by perform_pca, e.g., RANDOM_STATE = 42
# Ensure OUTPUT_DIR is defined globally if used by the calling code.

def prepare_embeddings_for_pca(embeddings_list: list[np.ndarray]) -> np.ndarray:
    """
    Prepares embeddings for PCA by concatenating them.
    """
    if not embeddings_list:
        raise ValueError("Embeddings list is empty. Cannot proceed.")
    
    concatenated_embeddings = np.concatenate(embeddings_list, axis=0)
    print(f"Shape of concatenated embeddings for PCA/Dimensionality Reduction: {concatenated_embeddings.shape}")
    return concatenated_embeddings

def perform_dimensionality_reduction(embeddings: np.ndarray, method="pca", n_components=2, random_state=42, **kwargs) -> np.ndarray:
    """
    Performs dimensionality reduction (PCA by default) on the given embeddings.
    """
    print(f"\n--- Performing {method.upper()} with {n_components} components ---")
    if method.lower() == "pca":
        operator = PCA(n_components=n_components, random_state=random_state)
    # Add elif for "phate", "umap", etc. if you integrate them later
    # elif method.lower() == "phate":
    #     from phate import PHATE
    #     operator = PHATE(n_components=n_components, random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
    reduced_embeddings = operator.fit_transform(embeddings)
    print(f"Shape of {n_components}D {method.upper()} embeddings: {reduced_embeddings.shape}")
    eigenvectors = operator.components_[:n_components] if hasattr(operator, 'components_') else None
    return reduced_embeddings, eigenvectors

def create_plot_dataframe(
    reduced_embeddings: np.ndarray, 
    all_labels: list[str], 
    all_positions: list[int] | torch.Tensor, # Union type for flexibility
    words_for_df: list[str]
) -> pd.DataFrame:
    """
    Creates a DataFrame for plotting from reduced embeddings and metadata.
    Detects dimensionality (2D or 3D) from reduced_embeddings shape.
    """
    if reduced_embeddings.shape[0] != len(all_labels):
        raise ValueError(f"Mismatch between number of embeddings ({reduced_embeddings.shape[0]}) and number of labels ({len(all_labels)})!")
    
    # Ensure positions_for_df is a Python list of ints
    if isinstance(all_positions, torch.Tensor):
        positions_for_df = all_positions.cpu().tolist()
    elif isinstance(all_positions, np.ndarray):
        positions_for_df = all_positions.tolist()
    else:
        positions_for_df = all_positions # Assuming it's already a list of ints

    num_components = reduced_embeddings.shape[1]
    df_data = {
        'PCA1': reduced_embeddings[:, 0],
        'PCA2': reduced_embeddings[:, 1],
        'label': all_labels,
        'word': words_for_df,
        'position': positions_for_df
    }
    if num_components >= 3:
        df_data['PCA3'] = reduced_embeddings[:, 2]
        print("Creating DataFrame for 3D plotting.")
    else:
        print("Creating DataFrame for 2D plotting.")
        
    df_plot = pd.DataFrame(df_data)
    print("DataFrame for plotting created successfully:")
    print(df_plot.head())
    return df_plot

# --- New Generic Plotting Function ---
def plot_dimensionality_reduction_results(
    df_plot: pd.DataFrame,
    words_in_plot: list[str],
    output_dir: str,
    base_plot_name: str, # e.g., "Final_Token_Embeddings" or "Initial_Token_Embeddings"
    reduction_method_label: str = "PCA", # e.g., "PCA", "PHATE"
    enable_interactive_text: bool = True
):
    """
    Generates interactive HTML and static PNG plots for 2D or 3D reduced embeddings.
    Determines dimensionality based on the presence of 'PCA3' column in df_plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    is_3d = 'PCA3' in df_plot.columns
    plot_dimensionality_str = "3D" if is_3d else "2D"
    
    # Common arguments for scatter functions
    scatter_args_common = {
        'color': 'position',
        'symbol': 'word',
        'hover_data': ['label', 'word', 'position']
    }
    
    scatter_args_interactive = scatter_args_common.copy()
    if enable_interactive_text:
        scatter_args_interactive['text'] = 'position'

    # Common layout arguments
    common_layout_args = {
        'width': 1200,
        'height': 900,
        'coloraxis_colorbar_title_text': 'Position',
        'legend_title_text': 'Word'
    }

    # --- Interactive Plot ---
    interactive_title = (
        f"{plot_dimensionality_str} {reduction_method_label}: {base_plot_name} by Position (Interactive) for {'_'.join(words_in_plot)}"
    )
    
    if is_3d:
        fig_interactive = px.scatter_3d(df_plot, x='PCA1', y='PCA2', z='PCA3', **scatter_args_interactive, title=interactive_title)
        fig_interactive.update_layout(
            **common_layout_args,
            scene=dict(xaxis_title=f'{reduction_method_label}1', yaxis_title=f'{reduction_method_label}2', zaxis_title=f'{reduction_method_label}3')
        )
        fig_interactive.update_traces(marker=dict(size=5), textposition='top center', textfont_size=8)
    else: # 2D
        fig_interactive = px.scatter(df_plot, x='PCA1', y='PCA2', **scatter_args_interactive, title=interactive_title)
        fig_interactive.update_layout(
            **common_layout_args,
            xaxis_title=f'{reduction_method_label}1', yaxis_title=f'{reduction_method_label}2',
            plot_bgcolor='rgba(240, 240, 240, 0.95)'
        )
        fig_interactive.update_traces(marker=dict(size=7), textposition='top center', textfont_size=8)
        fig_interactive.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')
        fig_interactive.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')

    interactive_filename = f"{'_'.join(words_in_plot)}_{base_plot_name}_{reduction_method_label}_{plot_dimensionality_str}_interactive.html"
    html_path = os.path.join(output_dir, interactive_filename)
    fig_interactive.write_html(html_path)
    print(f"Interactive {plot_dimensionality_str} {reduction_method_label} plot saved to: {html_path}")

    # --- Static Plot ---
    static_title = (
        f"{plot_dimensionality_str} {reduction_method_label}: {base_plot_name} by Position (Static) for {'_'.join(words_in_plot)}"
    )
    if is_3d:
        # For static 3D, text annotations are usually too cluttered.
        fig_static = px.scatter_3d(df_plot, x='PCA1', y='PCA2', z='PCA3', **scatter_args_common, title=static_title)
        fig_static.update_layout(
            **common_layout_args,
            scene=dict(xaxis_title=f'{reduction_method_label}1', yaxis_title=f'{reduction_method_label}2', zaxis_title=f'{reduction_method_label}3')
        )
        fig_static.update_traces(marker=dict(size=4))
    else: # 2D
        fig_static = px.scatter(df_plot, x='PCA1', y='PCA2', **scatter_args_common, title=static_title)
        fig_static.update_layout(
            **common_layout_args,
            xaxis_title=f'{reduction_method_label}1', yaxis_title=f'{reduction_method_label}2',
            plot_bgcolor='rgba(240, 240, 240, 0.95)'
        )
        fig_static.update_traces(marker=dict(size=6))
        fig_static.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')
        fig_static.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')
    
    static_filename_base = f"{'_'.join(words_in_plot)}_{base_plot_name}_{reduction_method_label}_{plot_dimensionality_str}_static"
    png_path = os.path.join(output_dir, f"{static_filename_base}.png")
    try:
        fig_static.write_image(png_path)
        print(f"Static {plot_dimensionality_str} {reduction_method_label} plot saved to: {png_path}")
    except ValueError as e:
        if "kaleido" in str(e).lower() or "indexOf is not a function" in str(e).lower() or "error code 525" in str(e).lower():
            print(f"Kaleido error for static {plot_dimensionality_str} {reduction_method_label} plot: {e}. Skipping PNG save.")
            print("Common fixes: ensure Kaleido is installed (`pip install -U kaleido plotly`), or simplify the figure.")
        else:
            print(f"An unexpected error occurred while writing static {plot_dimensionality_str} {reduction_method_label} plot: {e}")

#%%
# --- Process and Plot LAST Token Embeddings (Final Hidden States) ---
if last_token_embeddings_list:
    print("\n\n--- Processing LAST Token Embeddings (Final Hidden States) ---")
    embeddings_final_concatenated = prepare_embeddings_for_pca(last_token_embeddings_list)
    
    # For 2D PCA Plot
    reduced_embeddings_2d_final, last_token_eigenvectors = perform_dimensionality_reduction(
        embeddings_final_concatenated, method="pca", n_components=2, random_state=RANDOM_STATE
    )
    words_for_df_final = [label.split('_pos')[0] for label in all_labels] # Recalculate or pass if already available
    df_plot_2d_final = create_plot_dataframe(
        reduced_embeddings_2d_final, all_labels, all_positions, words_for_df_final
    )
    plot_dimensionality_reduction_results(
        df_plot_2d_final, words_to_test, OUTPUT_DIR, "Final_Token_Embeddings", "PCA"
    )

    # For 3D PCA Plot
    reduced_embeddings_3d_final, last_token_eigenvectors3d = perform_dimensionality_reduction(
        embeddings_final_concatenated, method="pca", n_components=3, random_state=RANDOM_STATE
    )
    df_plot_3d_final = create_plot_dataframe(
        reduced_embeddings_3d_final, all_labels, all_positions, words_for_df_final
    )
    plot_dimensionality_reduction_results(
        df_plot_3d_final, words_to_test, OUTPUT_DIR, "Final_Token_Embeddings", "PCA"
    )
else:
    print("last_token_embeddings_list is empty. Skipping PCA and plotting for final hidden states.")
#%%
# --- Process and Plot FIRST Token Embeddings (Initial E_token + PE_pos) ---
if first_token_embeddings_list: # Assuming you have this list from the modified pipeline
    print("\n\n--- Processing FIRST Token Embeddings (Initial E_token + PE_pos) ---")
    embeddings_initial_concatenated = prepare_embeddings_for_pca(first_token_embeddings_list)
    
    # For 2D PCA Plot
    reduced_embeddings_2d_initial, first_token_eigenvectors = perform_dimensionality_reduction(
        embeddings_initial_concatenated, method="pca", n_components=2, random_state=RANDOM_STATE
    )
    words_for_df_initial = [label.split('_pos')[0] for label in all_labels] # Recalculate or pass
    df_plot_2d_initial = create_plot_dataframe(
        reduced_embeddings_2d_initial, all_labels, all_positions, words_for_df_initial
    )
    plot_dimensionality_reduction_results(
        df_plot_2d_initial, words_to_test, OUTPUT_DIR, "Initial_Frontend_Embeddings", "PCA"
    )

    # For 3D PCA Plot
    reduced_embeddings_3d_initial, first_token_eigenvectors3d = perform_dimensionality_reduction(
        embeddings_initial_concatenated, method="pca", n_components=3, random_state=RANDOM_STATE
    )
    df_plot_3d_initial = create_plot_dataframe(
        reduced_embeddings_3d_initial, all_labels, all_positions, words_for_df_initial
    )
    plot_dimensionality_reduction_results(
        df_plot_3d_initial, words_to_test, OUTPUT_DIR, "Initial_Frontend_Embeddings", "PCA"
    )
else:
    print("first_token_embeddings_list is empty. Skipping PCA and plotting for initial frontend embeddings.")

# %%

most_important_eigenvector=last_token_eigenvectors[0]

# Print the most important eigenvector
print("Most important eigenvector (first component):")
print(most_important_eigenvector)
# Plot the most important eigenvector
plt.figure(figsize=(10, 6))
plt.bar(range(len(most_important_eigenvector)), most_important_eigenvector)
plt.title("Most Important Eigenvector (First Component)")
plt.xlabel("Feature Index")
plt.ylabel("Eigenvector Value")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "most_important_eigenvector.png"))
plt.show()
# %%
# Example position
ex_idx = 100
example_token_sequence = all_token_sequences[ex_idx]
example_position = all_positions[ex_idx]

# Print the example token sequence and its position
print(f"Example token sequence at index {ex_idx}: {example_token_sequence}")
print(f"Example position for the token sequence: {example_position}")
# Print the corresponding word
example_word = all_labels[ex_idx].split('_pos')[0]

example_out = text2vec.predict_from_token_ids(
    example_token_sequence, # Add batch dimension
    target_device=DEVICE
)

example_sentence_embedding, example_last_hidden_states, example_initial_embeds, example_padding_mask = example_out

# Print the shapes of the outputs
print(f"Shape of example sentence embedding: {example_sentence_embedding.shape}")
print(f"Shape of example last hidden states: {example_last_hidden_states.shape}")
print(f"Shape of example initial embeds: {example_initial_embeds.shape}")

# Print the specific token's last hidden state
# %% Load decoder
class EmbeddingToTokenModelPipeline(torch.nn.Module):
    model: SonarEncoderDecoderModel
    tokenizer: TextTokenizer
    device: Device
    def __init__(
        self,
        decoder: Union[str, ConditionalTransformerDecoderModel],
        tokenizer: Union[str, TextTokenizer],
        device: Device = CPU,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        Args:
            decoder (Union[str, ConditionalTransformerDecoderModel]): either card name or model object
            tokenizer (Union[str, TextTokenizer]): either card name or tokenizer object
            device (device, optional): Defaults to CPU.
            dtype (DataType, optional): The data type of the model parameters and buffers.

        """
        super().__init__()
        if isinstance(decoder, str):
            self.decoder = load_sonar_text_decoder_model(
                decoder, device=device, dtype=dtype, progress=False
            )
        else:
            self.decoder = decoder.to(device=device, dtype=dtype)
        if isinstance(tokenizer, str):
            tokenizer = load_sonar_tokenizer(tokenizer, progress=False)
        else:
            tokenizer = tokenizer
            
        encoder = DummyEncoderModel(self.decoder.model_dim).eval().to(DEVICE)  # type: ignore

        self.device = device

        self.model = SonarEncoderDecoderModel(encoder, self.decoder).eval()  # type: ignore

    @torch.inference_mode()
    def predict(
        self,
        input_embeddings: torch.Tensor,              # Shape: [batch_size, seq_len, hidden_dim]
        prompt_token_ids: torch.Tensor,              # e.g., torch.tensor([[ENG_LANG_ID1], [ENG_LANG_ID2]])
                                                     # Shape: [batch_size, prompt_len]
        max_new_tokens: int = 50,                    # Max new tokens to generate after prompt
        sampler: Optional[Sampler] = None,
        generator_kwargs: Optional[dict] = None,
        batch_size: int = 16,
        progress_bar: bool = False,

    ) -> List[List[int]]: # Returns a list (for batch) of lists of token IDs (top hypothesis)
        """
        Generates sequences of token IDs from input embeddings.

        :param input_embeddings:
            The embeddings to use as encoder output. Shape: (N, S_enc, M)
        :param input_padding_masks:
            The padding mask for ``input_embeddings``. Shape: (N, S_enc)
        :param prompt_token_ids:
            The initial token(s) to seed the decoder. Shape: (N, S_prm)
            (e.g., target language ID).
        :param max_new_tokens:
            Maximum number of new tokens to generate after the prompt.
        :param sampler:
            If provided, uses SamplingSeq2SeqGenerator. Otherwise, BeamSearch.
        :param generator_kwargs:
            Additional keyword arguments for the chosen Seq2SeqGenerator
            (e.g., beam_size, len_penalty for BeamSearch).

        :returns:
            A list where each element is a list of token IDs representing the
            top generated sequence for the corresponding input embedding.
            Includes prompt tokens if generator's echo_prompt=True (default for BeamSearch in fairseq2).
        """

        if sampler is not None:
            generator: Seq2SeqGenerator = SamplingSeq2SeqGenerator(
                self.model, sampler, **generator_kwargs
            )
        else:
            # Default to BeamSearch if no sampler provided
            generator = BeamSearchSeq2SeqGenerator(self.model, **generator_kwargs)

        # The generator's __call__ for BeamSearchSeq2SeqGenerator is:
        # (self, source_seqs, source_padding_mask, prompt_seqs, prompt_padding_mask)
        output_gen = generator(
            source_seqs=input_embeddings.to(self.device),             # Our embeddings act as "source_seqs" for DummyEncoder
            source_padding_mask=None,
            prompt_seqs=prompt_token_ids.to(self.device),
            prompt_padding_mask=None
        )

        return output_gen.hypotheses

#%%
vec2text = EmbeddingToTokenModelPipeline(
    decoder="text_sonar_basic_decoder",
    tokenizer="text_sonar_basic_encoder",  # Using the same tokenizer for simplicity
    device=DEVICE
)
#%%
with torch.no_grad():
    vec2text.predict(
        input_embeddings=example_sentence_embedding.unsqueeze(0),  # Add batch dimension
        prompt_token_ids=torch.tensor([[ENG_LANG_TOKEN_IDX]], device=DEVICE),  # Use the language ID as prompt
        max_new_tokens=50,  # Generate up to 50 new tokens
        sampler=None,  # Use default BeamSearch
        generator_kwargs={"echo_prompt": True}  # Include the prompt in the output
    )
# %%
