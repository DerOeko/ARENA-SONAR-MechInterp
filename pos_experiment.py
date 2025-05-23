#%%
import torch
import random
import itertools
import matplotlib.pyplot as plt 
import numpy as np
import os
from tqdm import tqdm

# SONAR and fairseq2 imports
from sonar.models.sonar_text import load_sonar_tokenizer
from sonar.models.encoder_model import SonarEncoderModel # For type hinting
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, extract_sequence_batch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask
from fairseq2.data import Collater 
from fairseq2.typing import Device, DataType, CPU
# Plotting
import plotly.express as px
import pandas as pd
from phate import PHATE

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
    """
    Custom pipeline that allows predicting embeddings from pre-tokenized token ID sequences.
    """
    def __init__(self, encoder, 
                 tokenizer, # TextTokenizer from fairseq2.data.text
                 device: Device = CPU, 
                 dtype = None):
        # Initialize the parent class
        # The parent __init__ loads the model and tokenizer if names are given
        super().__init__(encoder, tokenizer, device, dtype)
        # self.model is SonarEncoderModel, self.tokenizer is TextTokenizer

    @torch.inference_mode()
    def predict_from_token_ids(
        self,
        token_id_sequences: torch.Tensor, # Expects a 2D tensor [batch_size, seq_len]
        target_device = None,
    ) -> torch.Tensor:
        """
        Generates sentence embeddings from a batch of pre-constructed token ID sequences.
        """
        if not isinstance(token_id_sequences, torch.Tensor) or token_id_sequences.ndim > 2:
            raise ValueError("Input token_id_sequences must be a 2D torch.Tensor [batch_size, seq_len]")
        
        if token_id_sequences.ndim == 1:
            # If a single sequence is passed, add batch dimension
            token_id_sequences = token_id_sequences.unsqueeze(0)
        elif token_id_sequences.ndim != 2:
            raise ValueError("Input token_id_sequences must be a 2D tensor [batch_size, seq_len]")

        # Ensure the input tensor is on the same device as the model
        seqs = token_id_sequences.to(self.device)

        # 1. Create the PaddingMask
        # True for non-pad tokens, False for pad tokens.
        # The model's pad_idx is accessible via self.tokenizer.vocab_info.pad_idx
        padding_mask_bool_tensor = (seqs != self.tokenizer.vocab_info.pad_idx)
        # Calculate the true lengths of sequences (number of non-padding tokens)
        seq_lengths = padding_mask_bool_tensor.sum(dim=1)
        
        padding_mask = PaddingMask(seq_lengths, batch_seq_len=seqs.shape[1])
        # 2. Construct the batch dictionary as expected by SonarEncoderModel's forward pass
        # Based on fairseq2, this is typically a dictionary
        batch_input_for_model = SequenceBatch(
            seqs=seqs,
            padding_mask=padding_mask)
            
        # 3. Call the underlying SonarEncoderModel's forward method
        # self.model is an instance of SonarEncoderModel.
        # Its forward pass returns an object/dict that contains 'sentence_embeddings'.
        model_output = self.model(batch_input_for_model) 
        
        # 4. Extract the sentence embeddings
        # This relies on SonarEncoderModel's output structure.
        sentence_embeddings = model_output.sentence_embeddings 
        
        return sentence_embeddings.to(target_device or self.device)

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

words_to_test = ["dog"]
word_token_ids = {word: tokenizer_encoder(word)[1] for word in words_to_test}
print(f"Token IDs for words {words_to_test}: {word_token_ids}")

all_token_sequences = []
all_labels = []
for word_str in words_to_test:
    for i in range(1, MAX_SEQ_LEN - 1, MAX_SEQ_LEN // 10):
        token_ids = torch.full((MAX_SEQ_LEN,), PAD_IDX, dtype=torch.long, device= DEVICE)
        token_ids[0] = 256047
        token_ids[-1] = EOS_IDX
        token_ids[i] = word_token_ids[word_str]
        
        all_token_sequences.append(token_ids)
        all_labels.append(f"{word_str}_pos{i}")

# Convert to tensor
all_token_sequences = torch.stack(all_token_sequences).to(DEVICE)
#%% Get word embeddings
INFERENCE_BATCH_SIZE = 96
word_embeddings = []

with torch.no_grad():
    for i in tqdm(range(0, len(all_token_sequences), INFERENCE_BATCH_SIZE)):
        batch_tokens_ids = all_token_sequences[i:i + INFERENCE_BATCH_SIZE]
        batch_embeddings = text2vec.predict_from_token_ids(batch_tokens_ids)
        word_embeddings.append(batch_embeddings.cpu().numpy())

    print(f"Generated embeddings for words: {words_to_test}")

# %% --- PCA and Plotting ---
print("\n--- PCA and Plotting ---")
# Ensure these imports are at the top of your main script file
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import numpy as np
import os

# Assuming 'word_embeddings' is a list of numpy arrays (output of your batch processing)
# and 'all_labels' (e.g., ["dog_pos0", "dog_pos1", ...]) are defined from previous cells.

# Concatenate batch embeddings into a single array for PCA
embeddings_for_pca = np.concatenate(word_embeddings, axis=0)

pca_operator = PCA(n_components=2, random_state=RANDOM_STATE)
embeddings_2d_pca = pca_operator.fit_transform(embeddings_for_pca)

# Prepare data for DataFrame
words_for_df = [label.split('_pos')[0] for label in all_labels]
positions_for_df = [int(label.split('_pos')[1]) for label in all_labels]

df_pca_plot = pd.DataFrame({
    'PCA1': embeddings_2d_pca[:, 0],
    'PCA2': embeddings_2d_pca[:, 1],
    'label': all_labels,
    'word': words_for_df,
    'position': positions_for_df
})

# --- Interactive PCA Plot (for HTML) ---
# Points are colored by position, symbols distinguish words.
print("Generating interactive PCA plot (colored by position)...")
fig_pca_interactive = px.scatter(
    df_pca_plot,
    x='PCA1',
    y='PCA2',
    color='position',          # Color points by their position
    symbol='word',             # Different symbol for each word
    text='position',           # Display position number on the marker
    hover_data=['label', 'word', 'position'], # Info on hover
    title="PCA: Word Embeddings by Position (Interactive)"
)

fig_pca_interactive.update_traces(
    marker=dict(size=7), 
    textposition='top center', 
    textfont_size=8,
    selector=dict(mode='markers+text') # Ensure text is shown with markers
)
fig_pca_interactive.update_layout(
    xaxis_title='PCA Component 1',
    yaxis_title='PCA Component 2',
    plot_bgcolor='rgba(240, 240, 240, 0.95)',
    width=1200,
    height=900,
    coloraxis_colorbar_title_text='Position', # Label for the color bar
    legend_title_text='Word'
)
fig_pca_interactive.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')
fig_pca_interactive.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')

os.makedirs(OUTPUT_DIR, exist_ok=True) # OUTPUT_DIR should be defined earlier, e.g., "./data/"
html_pca_path = os.path.join(OUTPUT_DIR, "word_positional_embeddings_pca_interactive.html")
fig_pca_interactive.write_html(html_pca_path)
print(f"Interactive PCA plot saved to: {html_pca_path}")
# fig_pca_interactive.show() # Uncomment to display if running in a suitable environment

# --- Static PCA Plot (for PNG) ---
# Points colored by position, no text on markers for cleaner static image and Kaleido stability.
print("Generating static PCA plot for PNG export...")
fig_pca_static = px.scatter(
    df_pca_plot,
    x='PCA1',
    y='PCA2',
    color='position', # Color points by their position
    symbol='word',    # Different symbol for each word
    # NO 'text' argument here for the static image
    hover_data=['label', 'word', 'position'], # Ignored by write_image but good for consistency
    title='PCA: Word Embeddings by Position (Static)'
)

fig_pca_static.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
fig_pca_static.update_layout(
    xaxis_title='PCA Component 1',
    yaxis_title='PCA Component 2',
    plot_bgcolor='rgba(240, 240, 240, 0.95)',
    width=1200,
    height=900,
    coloraxis_colorbar_title_text='Position',
    legend_title_text='Word'
)
fig_pca_static.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')
fig_pca_static.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')

png_pca_path = os.path.join(OUTPUT_DIR, "word_positional_embeddings_pca_static.png")
try:
    fig_pca_static.write_image(png_pca_path)
    print(f"Static PCA plot saved to: {png_pca_path}")
except ValueError as e:
    if "kaleido" in str(e).lower() or "indexOf is not a function" in str(e).lower() or "error code 525" in str(e).lower():
        print(f"Kaleido error during static image generation for PCA plot: {e}. Skipping PNG save.")
        print("Common fixes: ensure Kaleido is installed (`pip install -U kaleido plotly`), or simplify the figure.")
    else:
        print(f"An unexpected error occurred while writing static PCA image: {e}")
#%%
import plotly.graph_objects as go
print("Attempting to save static PNG image...")
# Create a new figure for static export, removing or simplifying 'text'
fig_static = px.scatter(
    df_pca_test,
    x='PCA1',
    y='PCA2',
    color='word',
    symbol='word',
    hover_data=['label', 'position'], # Hover data is fine, it's ignored by write_image
    text='position', # OPTION 1: Show position number instead of full label for static image (might still be too much)
    title='PHATE: Word Positional Embeddings (Static)' # Ensure title is clear
)

# Re-add trajectory lines to the static figure if desired
if 'word_token_ids' in locals(): # Check if word_token_ids is available
    for word_str_val in word_token_ids.keys(): # Use the keys from your actual word_token_ids dict
        df_word = df_pca_test[df_pca_test['word'] == word_str_val].sort_values('position')
        if not df_word.empty:
            fig_static.add_trace(go.Scatter(
                x=df_word['PCA1'], y=df_word['PCA2'], mode='lines',
                line=dict(width=1), name=f'{word_str_val} trajectory (static)',
                legendgroup=word_str_val, showlegend=False
            ))

fig_static.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
fig_static.update_layout(
    xaxis_title='PCA1',
    yaxis_title='PCA2',
    plot_bgcolor='rgba(240, 240, 240, 0.95)',
    width=1200,
    height=900,
    legend_title_text='Word'
)
fig_static.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')
fig_static.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')

png_path = os.path.join(OUTPUT_DIR, "word_positional_embeddings_phate.png")
try:
    fig_static.write_image(png_path)
    print(f"Static PCA plot saved to: {png_path}")
except ValueError as e:
    if "kaleido" in str(e).lower() or "indexOf is not a function" in str(e).lower() or "error code 525" in str(e).lower():
        print(f"Kaleido error during static image generation. Skipping PNG save. Error: {e}")
        print("Common fixes: ensure Kaleido is installed (`pip install -U kaleido plotly`), or simplify the figure (e.g. by removing text labels on points).")
    else:
        print(f"An unexpected error occurred while writing static image: {e}")
# Re-add trajectory lines to the static figure if desired
if 'word_token_ids' in locals(): # Check if word_token_ids is available
    for word_str_val in word_token_ids.keys(): # Use the keys from your actual word_token_ids dict
        df_word = df_pca_test[df_pca_test['word'] == word_str_val].sort_values('position')
        if not df_word.empty:
            fig_static.add_trace(go.Scatter(
                x=df_word['PCA1'], y=df_word['PCA1'], mode='lines',
                line=dict(width=1), name=f'{word_str_val} trajectory (static)',
                legendgroup=word_str_val, showlegend=False
            ))

fig_static.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
fig_static.update_layout(
    xaxis_title='PCA1',
    yaxis_title='PCA2',
    plot_bgcolor='rgba(240, 240, 240, 0.95)',
    width=1200,
    height=900,
    legend_title_text='Word'
)
fig_static.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')
fig_static.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')

png_path = os.path.join(OUTPUT_DIR, "word_positional_embeddings_phate.png")
try:
    fig_static.write_image(png_path)
    print(f"Static PCA plot saved to: {png_path}")
except ValueError as e:
    if "kaleido" in str(e).lower() or "indexOf is not a function" in str(e).lower() or "error code 525" in str(e).lower():
        print(f"Kaleido error during static image generation. Skipping PNG save. Error: {e}")
        print("Common fixes: ensure Kaleido is installed (`pip install -U kaleido plotly`), or simplify the figure (e.g. by removing text labels on points).")
    else:
        print(f"An unexpected error occurred while writing static image: {e}")

#%%
phate_operator = PHATE(n_components=2, random_state = RANDOM_STATE, n_jobs=-1, verbose=1)
embeddings_2d = phate_operator.fit_transform(np.concatenate(word_embeddings, axis=0))

#%%
# Create a DataFrame for Plotly
words_for_df = [label.split('_pos')[0] for label in all_labels]
positions_for_df = [int(label.split('_pos')[1]) for label in all_labels]

df_phate = pd.DataFrame({
    'PHATE1': embeddings_2d[:, 0],
    'PHATE2': embeddings_2d[:, 1],
    'label': all_labels,
    'word': words_for_df,
    'position': positions_for_df
})

print("Plotting PHATE results...")
# Create interactive plot with plotly

fig = px.scatter(
    df_phate,
    x='PHATE1',
    y='PHATE2',
    color='word',
    symbol='word',
    hover_data=['label', 'position'], # Text will appear on hover in HTML
    text='position', # OPTION 1: Show position number instead of full label for static image (might still be too much)
    # If using text='position', ensure it's concise.
)

fig.update_layout(
    xaxis_title='PHATE Component 1',
    yaxis_title='PHATE Component 2',
    plot_bgcolor='rgba(240, 240, 240, 0.95)',
    width=1200,
    height=900,
    legend_title_text='Word'
)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True) # OUTPUT_DIR was defined as "./data/"
html_path = os.path.join(OUTPUT_DIR, "word_positional_embeddings_phate.html")
png_path = os.path.join(OUTPUT_DIR, "word_positional_embeddings_phate.png")

fig.write_html(html_path)
print(f"\nInteractive PHATE plot saved to: {html_path}")
#%%
# Add trajectory lines (this part is good)
for word_str_val in word_token_ids.keys(): # Iterate using the keys from your word_token_ids
    df_word = df_phate[df_phate['word'] == word_str_val].sort_values('position')
    if not df_word.empty:
        fig.add_trace(go.Scatter(
            x=df_word['PHATE1'], 
            y=df_word['PHATE2'],
            mode='lines',
            line=dict(width=1),
            name=f'{word_str_val} trajectory',
            legendgroup=word_str_val,
            showlegend=False
        ))

fig.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
# If you used text='position' above and it's still too cluttered for the static image:
# fig.update_traces(textposition='top center', textfont=dict(size=8), selector=dict(mode='markers'))


fig.update_layout(
    xaxis_title='PHATE Component 1',
    yaxis_title='PHATE Component 2',
    plot_bgcolor='rgba(240, 240, 240, 0.95)',
    width=1200,
    height=900,
    legend_title_text='Word'
)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True) # OUTPUT_DIR was defined as "./data/"
html_path = os.path.join(OUTPUT_DIR, "word_positional_embeddings_phate.html")
png_path = os.path.join(OUTPUT_DIR, "word_positional_embeddings_phate.png")

fig.write_html(html_path)
print(f"\nInteractive PHATE plot saved to: {html_path}")

# Create a version of the figure specifically for static image export, without text labels on points
fig_for_static = px.scatter(
    df_phate,
    x='PHATE1',
    y='PHATE2',
    color='word',
    symbol='word',
    hover_data=['label', 'position'], # Keep hover for potential use, though not for static
    title=f'PHATE of Single Word Embeddings by Position (Max Pos: {NUM_POSITIONS_TO_TEST_PER_WORD-1})'
)
for word_str_val in word_token_ids.keys():
    df_word = df_phate[df_phate['word'] == word_str_val].sort_values('position')
    if not df_word.empty:
        fig_for_static.add_trace(go.Scatter(
            x=df_word['PHATE1'], y=df_word['PHATE2'],
            mode='lines', line=dict(width=1), name=f'{word_str_val} trajectory',
            legendgroup=word_str_val, showlegend=False
        ))
fig_for_static.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
fig_for_static.update_layout(
    xaxis_title='PHATE Component 1', yaxis_title='PHATE Component 2',
    plot_bgcolor='rgba(240, 240, 240, 0.95)', width=1200, height=900, legend_title_text='Word'
)
fig_for_static.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')
fig_for_static.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')


try:
    fig_for_static.write_image(png_path) # Use the figure without text annotations
    print(f"Static PHATE plot saved to: {png_path}")
except ValueError as e:
    if "kaleido" in str(e).lower() or "indexOf is not a function" in str(e) or "Error 525" in str(e): # Catching the 525
        print(f"Kaleido error during image generation. Skipping PNG save. Error: {e}")
        print("To save PNGs, ensure Kaleido is installed and working: pip install -U kaleido plotly")
        print("If Kaleido is installed, the figure might be too complex even without text. Try reducing points for static image or saving as SVG.")
    else:
        print(f"An unexpected error occurred while writing image: {e}")

fig.show() # Show the original interactive figure

print("\nAnalysis complete.")
#%%
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

#%%
class CustomTextToEmbeddingPipeline(TextToEmbeddingModelPipeline):
    @torch.inference_mode()
    def predict_from_token_ids(
        self,
        token_id_sequences: torch.Tensor, # Expects a 2D tensor [batch, seq_len]
        target_device = None
    ) -> torch.Tensor:
        """
        Generates sentence embeddings from a batch of pre-constructed token ID sequences.
        Assumes sequences are already on the correct device and include lang_id, word_id, eos_id, and padding.
        """
        if token_id_sequences.ndim != 2:
            raise ValueError("Input token_id_sequences must be a 2D tensor [batch_size, seq_len]")

        seqs = token_id_sequences.to(self.device) # Ensure it's on model's device

        # Create padding mask: True for non-pad tokens, False for pad tokens.
        # The model's internal pad_idx is self.tokenizer.vocab_info.pad_idx
        padding_mask_tensor = (seqs != self.tokenizer.vocab_info.pad_idx)
        padding_mask = PaddingMask(padding_mask_tensor, torch.sum(padding_mask_tensor, dim=1))

        # Construct the batch dictionary as expected by the SonarEncoderModel's forward pass
        batch = {"seqs": seqs, "padding_mask": padding_mask}
        
        # Call the underlying SonarEncoderModel's forward method
        # This typically returns an object or dict containing 'sentence_embeddings'
        # and/or 'encoder_output' from which sentence_embeddings are derived (e.g. by pooling).
        # The original pipeline does: model_output = self.model(batch_dict) -> model_output.sentence_embeddings
        # SonarEncoderModel.forward returns (encoder_output, padding_mask)
        # The sentence_embeddings attribute is added to the output by the pipeline's map function AFTER self.model call.
        # So, we need to replicate the pooling.
        encoder_output, _ = self.model.model(batch) # self.model.model is SonarTextTransformerEncoderModel
        
        # Standard SONAR pooling: take the embedding of the first token (lang_id)
        sentence_embeddings = encoder_output[:, 0, :] 
        
        return sentence_embeddings.to(target_device or self.device)

    # Override predict to dispatch
    @torch.inference_mode()
    def predict(
        self,
        input_data: Union[Path, Sequence[str], torch.Tensor], # Simplified for this use case
        source_lang: Optional[str] = "eng_Latn",
        batch_size: int = 32,
        max_seq_len: Optional[int] = None,
        progress_bar: bool = False,
        target_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if isinstance(input_data, torch.Tensor):
            # This means we're passing a batch of token_id sequences
            return self.predict_from_token_ids(
                input_data,
                target_device=target_device
            )
        elif isinstance(input_data, (Path, str)) or \
             (isinstance(input_data, list) and len(input_data) > 0 and isinstance(input_data[0], str)):
            # Input is string(s) or Path, use original logic
            if source_lang is None:
                raise ValueError("source_lang must be provided for string inputs.")
            return super().predict(
                input_data, # type: ignore
                source_lang=source_lang,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                progress_bar=progress_bar,
                target_device=target_device
            )
        else:
            raise TypeError(
                "Input must be a Path, a string, a sequence of strings, "
                "or a 2D Tensor of token IDs."
            )
#%% Change TextToEmbeddingModelPipeline to accept tokens directly
class TextToEmbeddingModelPipelineWithTokens(TextToEmbeddingModelPipeline):
    # ALWAYS GIVE A TENSOR TO THE TOKENIZER
    @torch.inference_mode()
    def predict(
        self,
        input,
        source_lang: str,
        batch_size: int = 5,
        max_seq_len,
        progress_bar: bool = False,
        target_device,
    ) -> torch.Tensor:
        """
        Transform the input texts (from a list of strings or from a text file) into a matrix of their embeddings.
        The texts are truncated to `max_seq_len` tokens,
        or, if it is not specified, to the maximum that the model supports.
        """
        model_max_len = self.model.encoder_frontend.pos_encoder.max_seq_len
        if max_seq_len is None:
            max_seq_len = model_max_len
        elif max_seq_len > model_max_len:
            raise ValueError(
                f"max_seq_len cannot be larger than max_seq_len of the encoder model: {model_max_len}"
            )

        n_truncated = 0

        def truncate(x: torch.Tensor) -> torch.Tensor:
            if x.shape[0] > max_seq_len:
                nonlocal n_truncated
                n_truncated += 1
            return x[:max_seq_len]

        pipeline: Iterable = (
            (
                read_text(Path(input))
                if isinstance(input, (str, Path))
                else read_sequence(input)
            )
            .map(tokenizer_encoder)
            .map(truncate)
            .bucket(batch_size)
            .map(Collater(self.tokenizer.vocab_info.pad_idx))
            .map(lambda x: extract_sequence_batch(x, self.device))
            .map(self.model)
            .map(lambda x: x.sentence_embeddings.to(target_device or self.device))
            .and_return()
        )
        if progress_bar:
            pipeline = add_progress_bar(pipeline, inputs=input, batch_size=batch_size)
        results: List[torch.Tensor] = list(iter(pipeline))

        if n_truncated:
            warnings.warn(
                f"For {n_truncated} input tensors for SONAR text encoder, "
                f"the length was truncated to {max_seq_len} elements."
            )

        sentence_embeddings = torch.cat(results, dim=0)
        return sentence_embeddings
#%% Load Sonar encoder and generate embeddings for tokenized sentences
print("\n--- Generating Embeddings for Sentences ---")

orig_tokenizer = load_sonar_tokenizer("text_sonar_basic_encoder")
vocab_info = orig_tokenizer.vocab_info
pad_idx = vocab_info.pad_idx
eos_idx = vocab_info.eos_idx

tokenizer_encoder = load_tokenizer()
tokenizer_decoder = orig_tokenizer.create_decoder()

words_to_test = ["dog", "cat", "car", "house", "tree"]
word_token_ids = {word: tokenizer_encoder.encode(word)[1] for word in words_to_test}
print(f"Token IDs for words {words_to_test}: {word_token_ids}")

#%%
# Load the Sonar encoder model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# check if local context has text2vec already loaded
if not hasattr(locals(), 'text2vec'):
    text2vec_withtokens = TextToEmbeddingModelPipelineWithTokens(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device
    )
#%%
max_seq_len = text2vec_withtokens._modules['model'].encoder_frontend.pos_encoder.max_seq_len

pad_str = tokenizer_decoder(torch.tensor([pad_idx]))
all_token_sequences = []
all_labels = []
eng_lang_token_idx = 256047
for word_str in words_to_test:
    for i in range(1, max_seq_len - 1):
        token_ids = torch.full((max_seq_len,), pad_idx, dtype=torch.long, device= device)
        token_ids[0] = 256047
        token_ids[-1] = eos_idx
        token_ids[i] = word_token_ids[word_str]
        
        all_token_sequences.append(token_ids)
        all_labels.append(f"{word_str}_pos{i}")

# Convert to tensor
all_token_sequences = torch.stack(all_token_sequences).to(device)
#%% Get word embeddings

with torch.no_grad():
    word_embeddings = text2vec_withtokens.predict_with_tokens(all_token_sequences, "eng_Latn")
    word_embeddings = word_embeddings.cpu().numpy()
    print(f"Generated embeddings for words: {words_to_test}")
    
# %% --- PCA and Plotting --- (Corrected PHATE section)
print("\n--- PHATE and Plotting ---") # Changed from PCA to PHATE

# Assuming final_embeddings_array and all_labels are correctly generated from previous cells

phate_operator = PHATE(n_components=2, random_state=42, n_jobs=-1, verbose=1)
embeddings_2d_phate = phate_operator.fit_transform(word_embeddings)
#%%
# Create a DataFrame for Plotly
words_for_df = [label.split('_pos')[0] for label in all_labels]
positions_for_df = [int(label.split('_pos')[1]) for label in all_labels]

df_phate = pd.DataFrame({
    'PHATE1': embeddings_2d_phate[:, 0],
    'PHATE2': embeddings_2d_phate[:, 1],
    'label': all_labels,
    'word': words_for_df,
    'position': positions_for_df
})

print("Plotting PHATE results...")

fig = px.scatter(
    df_phate,
    x='PHATE1',
    y='PHATE2',
    color='word',
    symbol='word',
    hover_data=['label', 'position'], # Text will appear on hover in HTML
    # text='position', # OPTION 1: Show position number instead of full label for static image (might still be too much)
    # If using text='position', ensure it's concise.
    # OPTION 2 (Recommended for static image): Remove 'text' argument entirely for fig.write_image
    title=f'PHATE of Single Word Embeddings by Position (Max Pos: {NUM_POSITIONS_TO_TEST_PER_WORD-1})'
)

# Add trajectory lines (this part is good)
for word_str_val in word_token_ids.keys(): # Iterate using the keys from your word_token_ids
    df_word = df_phate[df_phate['word'] == word_str_val].sort_values('position')
    if not df_word.empty:
        fig.add_trace(go.Scatter(
            x=df_word['PHATE1'], 
            y=df_word['PHATE2'],
            mode='lines',
            line=dict(width=1),
            name=f'{word_str_val} trajectory',
            legendgroup=word_str_val,
            showlegend=False
        ))

fig.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
# If you used text='position' above and it's still too cluttered for the static image:
# fig.update_traces(textposition='top center', textfont=dict(size=8), selector=dict(mode='markers'))


fig.update_layout(
    xaxis_title='PHATE Component 1',
    yaxis_title='PHATE Component 2',
    plot_bgcolor='rgba(240, 240, 240, 0.95)',
    width=1200,
    height=900,
    legend_title_text='Word'
)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True) # OUTPUT_DIR was defined as "./data/"
html_path = os.path.join(OUTPUT_DIR, "word_positional_embeddings_phate.html")
png_path = os.path.join(OUTPUT_DIR, "word_positional_embeddings_phate.png")

fig.write_html(html_path)
print(f"\nInteractive PHATE plot saved to: {html_path}")

# Create a version of the figure specifically for static image export, without text labels on points
fig_for_static = px.scatter(
    df_phate,
    x='PHATE1',
    y='PHATE2',
    color='word',
    symbol='word',
    hover_data=['label', 'position'], # Keep hover for potential use, though not for static
    title=f'PHATE of Single Word Embeddings by Position (Max Pos: {NUM_POSITIONS_TO_TEST_PER_WORD-1})'
)
for word_str_val in word_token_ids.keys():
    df_word = df_phate[df_phate['word'] == word_str_val].sort_values('position')
    if not df_word.empty:
        fig_for_static.add_trace(go.Scatter(
            x=df_word['PHATE1'], y=df_word['PHATE2'],
            mode='lines', line=dict(width=1), name=f'{word_str_val} trajectory',
            legendgroup=word_str_val, showlegend=False
        ))
fig_for_static.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
fig_for_static.update_layout(
    xaxis_title='PHATE Component 1', yaxis_title='PHATE Component 2',
    plot_bgcolor='rgba(240, 240, 240, 0.95)', width=1200, height=900, legend_title_text='Word'
)
fig_for_static.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')
fig_for_static.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')


try:
    fig_for_static.write_image(png_path) # Use the figure without text annotations
    print(f"Static PHATE plot saved to: {png_path}")
except ValueError as e:
    if "kaleido" in str(e).lower() or "indexOf is not a function" in str(e) or "Error 525" in str(e): # Catching the 525
        print(f"Kaleido error during image generation. Skipping PNG save. Error: {e}")
        print("To save PNGs, ensure Kaleido is installed and working: pip install -U kaleido plotly")
        print("If Kaleido is installed, the figure might be too complex even without text. Try reducing points for static image or saving as SVG.")
    else:
        print(f"An unexpected error occurred while writing image: {e}")

fig.show() # Show the original interactive figure

print("\nAnalysis complete.")

#%%
# Visualize word embeddings using PCA


phate_operator = PHATE(n_components=2, random_state=42)
embeddings_2d = phate_operator.fit_transform(word_embeddings)

# Create a DataFrame for plotly
df = pd.DataFrame({
    'PHATE1': embeddings_2d[:, 0],
    'PHATE2': embeddings_2d[:, 1],
    'sentence': all_labels
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


#%%
# get different words

for i in range(max_seq_len):
    
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