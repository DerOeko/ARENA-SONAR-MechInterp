#%%
import os
import random
from typing import List

import numpy as np
import torch
from fairseq2.nn.padding import PaddingMask
from fairseq2.generation import BeamSearchSeq2SeqGenerator, TextTranslator
from fairseq2.models.sequence import SequenceBatch
from fairseq2.typing import CPU, DataType, Device
from sonar.inference_pipelines.text import TextToTextModelPipeline

# SONAR and fairseq2 imports
from sonar.models.sonar_text import load_sonar_tokenizer
from tqdm import tqdm

# --- Configuration (ensure these are consistent with your previous setup) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
MODEL_NAME_ENCODER = "text_sonar_basic_encoder" # Used for tokenizer info initially
MODEL_NAME_DECODER_TOKENIZER = "text_sonar_basic_decoder" # For the T2T pipeline's tokenizer and decoder model
# Using a smaller set for demonstration
# WORDS_TO_TEST = ["dog", "cat"] # Reduced for brevity in output
OUTPUT_DIR = "./data/" # Ensure this exists if you save anything
# Seed for reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(RANDOM_STATE)
# %% --- Tokenizer and Special Token IDs (from text_sonar_basic_encoder for consistency) ---
print("--- Initializing Tokenizer and Special IDs (using encoder's tokenizer for reference) ---")
# We use the 'encoder' model's tokenizer to get consistent special token IDs like PAD_IDX, EOS_IDX
# The T2T pipeline will use its own tokenizer ('text_sonar_basic_decoder') for processing.
# It's important that these tokenizers share the same special token IDs if we want to
# perfectly reconstruct inputs that use those IDs.
# However, for this test, we are constructing strings and letting the T2T pipeline's tokenizer handle them.
# Load a tokenizer to get vocab info (e.g., from the encoder model)
ref_tokenizer_for_special_ids = load_sonar_tokenizer(MODEL_NAME_ENCODER)
tokenizer_encoder_ref = ref_tokenizer_for_special_ids.create_encoder() # To get ENG_LANG_TOKEN_IDX
tokenizer_decoder_ref = ref_tokenizer_for_special_ids.create_decoder() # To decode special token IDs to strings
VOCAB_INFO_REF = ref_tokenizer_for_special_ids.vocab_info
PAD_IDX = VOCAB_INFO_REF.pad_idx
EOS_IDX = VOCAB_INFO_REF.eos_idx
UNK_IDX = VOCAB_INFO_REF.unk_idx
# UNK_IDX = VOCAB_INFO_REF.unk_idx # Not used in this specific sequence construction
# Get English Language Token ID (numeric)
dummy_tokenized_for_special_tokens = tokenizer_encoder_ref("test")
ENG_LANG_TOKEN_IDX_NUMERIC = dummy_tokenized_for_special_tokens[0].item()
# Get string representations of special tokens using the reference decoder
str_pad_token = tokenizer_decoder_ref(torch.tensor([PAD_IDX]))
str_eos_token = tokenizer_decoder_ref(torch.tensor([EOS_IDX]))
str_unk_token = tokenizer_decoder_ref(torch.tensor([UNK_IDX]))
str_eng_lang_token = tokenizer_decoder_ref(torch.tensor([ENG_LANG_TOKEN_IDX_NUMERIC])) # e.g., "eng_Latn"
print(f"Reference Language ID (numeric): {ENG_LANG_TOKEN_IDX_NUMERIC} -> String: '{str_eng_lang_token}'")
print(f"Reference PAD ID: {PAD_IDX} -> String: '{str_pad_token}'")
print(f"Reference EOS ID: {EOS_IDX} -> String: '{str_eos_token}'")
# Define MAX_SEQ_LEN (e.g., from one of the models, typically the encoder)
# This needs to be obtained from the actual model used if not hardcoded
# For demonstration, let's assume a MAX_SEQ_LEN.
# In a real scenario, you'd load the encoder model to get its pos_encoder.max_seq_len
# from sonar.models.encoder_model import SonarEncoderModel
# temp_encoder_model = SonarEncoderModel.from_pretrained(MODEL_NAME_ENCODER, device=DEVICE, dtype=torch.float32)
# MAX_SEQ_LEN = temp_encoder_model.encoder_frontend.pos_encoder.max_seq_len
# del temp_encoder_model
# For now, let's use a common value, but be aware this should match your model
MAX_SEQ_LEN = 514 # Or get it dynamically as shown above
print(f"Using MAX_SEQ_LEN: {MAX_SEQ_LEN}")
# %% --- Generate String Token Sequences ---
# all_string_sequences = []
# all_string_labels = []
# # all_string_positions = [] # Not strictly needed for this part but good for reference
# print(f"\n--- Generating String Token Sequences (Stepped Positions) ---")
# # for word_str in WORDS_TO_TEST:
# for word_str in WORDS_TO_TEST:
#     print(f"Preparing sequences for word: '{word_str}'")
#     # Using a step to reduce the number of sequences for demonstration
#     for i in range(1, MAX_SEQ_LEN - 1, max(1, MAX_SEQ_LEN // 10)): # Ensure step is at least 1
#         string_sequence = [str_pad_token] * MAX_SEQ_LEN
#         string_sequence[0] = str_eng_lang_token
#         string_sequence[MAX_SEQ_LEN - 1] = str_eos_token
#         string_sequence[i] = word_str
#         all_string_sequences.append(string_sequence)
#         all_string_labels.append(f"{word_str}_pos{i}")
#         # all_string_positions.append(i)
# print(f"Generated {len(all_string_sequences)} string sequences to test.")

# %%

class TextToTextModelPipelineWithTokenInput(TextToTextModelPipeline):

    @torch.inference_mode()
    def predict_from_token_ids(
        self,
        token_id_sequences: torch.Tensor,  # [batch, seq_len]
        source_lang: str,
        target_lang: str,
        batch_size: int = 5,
        progress_bar: bool = False,
        **generator_kwargs,
    ):
        # Ensure 2D tensor
        if token_id_sequences.ndim == 1:
            token_id_sequences = token_id_sequences.unsqueeze(0)
        assert token_id_sequences.ndim == 2

        pad_idx = self.tokenizer.vocab_info.pad_idx
        padding_mask_tensor = (token_id_sequences != pad_idx).to(torch.bool).to(token_id_sequences.device)
        padding_mask = PaddingMask(padding_mask_tensor, batch_seq_len=token_id_sequences.size(1))
        batch = SequenceBatch(token_id_sequences, padding_mask)

        generator_kwargs = generator_kwargs or {}
        model_max_seq_len = self.model.decoder.decoder_frontend.pos_encoder.max_seq_len
        generator_kwargs["max_seq_len"] = min(
            model_max_seq_len, generator_kwargs.get("max_seq_len", model_max_seq_len)
        )

        generator = BeamSearchSeq2SeqGenerator(self.model, **generator_kwargs)
        # Instead of using TextTranslator, use the generator directly to get token IDs
        # The generator expects input as a SequenceBatch
        # The output is a tuple (tokens, output_obj)
        # We'll collect the tokens for each item in the batch
        output = generator(
            batch.seqs,
            batch.padding_mask,
            None,
            None
        )
        return [hyps[0].seq for hyps in output.hypotheses]


# %% --- Initialize TextToTextModelPipeline ---
print("\n--- Initializing TextToTextModelPipeline ---")

text2text_pipeline = TextToTextModelPipelineWithTokenInput(
    encoder=MODEL_NAME_ENCODER,         # Or your specific encoder model name
    decoder=MODEL_NAME_DECODER_TOKENIZER, # Or your specific decoder model name
    tokenizer=MODEL_NAME_DECODER_TOKENIZER, # Tokenizer for the T2T pipeline
    device=DEVICE
)
print("TextToTextModelPipeline initialized successfully.")

# %% --- Test with TextToTextModelPipeline and Check Equivalence ---
# print("\n--- Running Equivalence Check through TextToTextModelPipeline ---")
# target_lang_str_for_pipeline = str_eng_lang_token
# print(f"Using target language for pipeline: '{target_lang_str_for_pipeline}'")
# equivalent_count = 0
# max_sequences_to_check = 5 # Limit for brevity in output
# for idx, string_sequence in enumerate(tqdm(all_string_sequences[:max_sequences_to_check], desc="Testing Sequences")):
#     current_label = all_string_labels[idx]
#     input_sentence_str = " ".join(string_sequence)
#     # The pipeline expects a list of source sentences and a list of target language IDs/strings
#     try:
#         # The predict method returns a list of output strings
#         output_sentences = text2text_pipeline.predict(
#             source_sentences=[input_sentence_str],
#             target_lang_ids=[target_lang_str_for_pipeline] # Must be a list
#         )
#         output_sentence_str = output_sentences[0] # Get the first (and only) output
#         is_equivalent = (input_sentence_str == output_sentence_str)
#         if is_equivalent:
#             equivalent_count += 1
#         print(f"\n--- Sequence: {current_label} ---")
#         print(f"Input Sentence (Joined Strings):\n\"{input_sentence_str[:100]}...\" (length: {len(input_sentence_str)})") # Print snippet
#         print(f"Model Output Sentence:\n\"{output_sentence_str[:100]}...\" (length: {len(output_sentence_str)})") # Print snippet
#         print(f"Are they equivalent? {'YES' if is_equivalent else 'NO'}")
#         if not is_equivalent:
#             # You could add more detailed diffing here if needed
#             pass
#     except Exception as e:
#         print(f"\nError processing sequence {current_label}: {e}")
#         print(f"Input was: \"{input_sentence_str[:100]}...\"")
# print(f"\n--- Equivalence Check Summary ---")
# print(f"Total sequences checked: {min(max_sequences_to_check, len(all_string_sequences))}")
# print(f"Number of equivalent sequences: {equivalent_count}")
# if min(max_sequences_to_check, len(all_string_sequences)) > 0:
#     accuracy = (equivalent_count / min(max_sequences_to_check, len(all_string_sequences))) * 100
#     print(f"Equivalence Accuracy: {accuracy:.2f}%")

# %%
test_sequence = ["dog", str_unk_token, str_unk_token, str_unk_token, str_unk_token]
tokenizer = text2text_pipeline.tokenizer.create_raw_encoder()

batch = torch.tensor(
    [
        [
            tokenizer(token) for token in test_sequence
        ]
    ],
).to(DEVICE)
print(batch)

# %%
preds = text2text_pipeline.predict_from_token_ids(batch, source_lang="eng_Latn", target_lang="eng_Latn")
# %%
preds
# %%
