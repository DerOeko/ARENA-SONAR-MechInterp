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
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import numpy as np
import os
import torch
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from tqdm.auto import tqdm

from src.utils.pca_utils import (
    perform_dimensionality_reduction,
    create_plot_dataframe
)
from sonar.models.sonar_text import (
    load_sonar_tokenizer,
)
from src.utils.chess_utils import (
    make_move_illegal,
    make_sequence_illegal,
    generate_legal_chess_sequences,
    make_one_san_move_illegal_by_random_destination,
    generate_illegal_game_sequences,
    generate_data_for_probing
)

# SONAR and fairseq2 imports
from sonar.models.sonar_text import load_sonar_tokenizer
# Plotting
import plotly.express as px
from src.custom_sonar_pipeline import CustomTextToEmbeddingPipeline
import numpy as np
import chess

def generate_data_for_probing(
    num_games_to_process: int,
    target_ply_lengths: list[int], # e.g., [10, 20] for 5-move and 10-move game prefixes
    # --- Pass necessary components for embedding ---
    token2vec_pipeline, 
    tokenizer_encoder_func,
    model_max_seq_len_val: int,
    pad_idx_val: int,
    device_val,
    # --- Other params ---
    max_moves_in_full_game: int = 30 # How long the base games should be
):
    """
    Generates game prefixes (text), their SONAR embeddings, 
    and corresponding board state features.
    """
    
    all_prefix_san_strings = []
    all_extracted_board_features = [] # List of dictionaries

    #print(f"Generating game data for probing at ply lengths: {target_ply_lengths}...")

    # 1. Generate Legal Game Sequences (long enough to get desired prefixes)
    # We need full games to extract prefixes from.
    # Make sure max_moves_in_full_game is at least max(target_ply_lengths)
    if not target_ply_lengths:
        print("Warning: No target_ply_lengths specified for prefixes.")
        return np.array([]), []
        
    actual_max_ply = max(target_ply_lengths)
    if max_moves_in_full_game < actual_max_ply // 2 + 1 : # ensure enough full moves
        max_moves_in_full_game = actual_max_ply // 2 + 5


    #print(f"Generating {num_games_to_process} base games up to {max_moves_in_full_game} moves each...")
    base_legal_games_san = generate_legal_chess_sequences(
        num_sequences=num_games_to_process,
        max_moves_per_sequence=max_moves_in_full_game, # Generate reasonably long games
        use_san=True
    )

    # 2. For each game, extract prefixes and their board states
    #print(f"Extracting prefixes and board features...")
    for game_san_str in tqdm(base_legal_games_san, desc="Processing Games for Prefixes"):
        moves_in_game = game_san_str.split()
        board = chess.Board()
        current_prefix_moves_san = []

        for ply_idx, move_san in enumerate(moves_in_game):
            try:
                # It's good to parse SAN to ensure it's valid and to push the move object
                # board.push_san(move_san) is quicker if SAN is guaranteed good
                parsed_move = board.parse_san(move_san)
                board.push(parsed_move)
                current_prefix_moves_san.append(move_san) # Use the original SAN string for the prefix
            except ValueError: # Illegal SAN move encountered
                # print(f"Warning: Skipping illegal SAN move '{move_san}' in sequence '{game_san_str}'")
                break # Stop processing this game if an illegal move is found

            current_ply_count = ply_idx + 1 # Ply count after this move

            if current_ply_count in target_ply_lengths:
                prefix_string_for_embedding = " ".join(current_prefix_moves_san)
                all_prefix_san_strings.append(prefix_string_for_embedding)

                # Extract features from the current 'board' state
                features = {}
                # Example 1: Is white Queen on the board?
                wq_squares = board.pieces(chess.QUEEN, chess.WHITE)
                features["wq_on_board"] = 1 if wq_squares else 0
                
                # Example 2: Is white King on g1?
                wk_square = board.king(chess.WHITE)
                features["wk_on_g1"] = 1 if wk_square == chess.G1 else 0
                
                # Example 3: Is e4 occupied by a white piece?
                piece_on_e4 = board.piece_at(chess.E4)
                features["e4_white_occupied"] = 1 if (piece_on_e4 and piece_on_e4.color == chess.WHITE) else 0
                
                # Example 4: Material difference
                mat_diff = 0
                piece_values = { chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9 }
                for piece_type_val, val in piece_values.items():
                    mat_diff += len(board.pieces(piece_type_val, chess.WHITE)) * val
                    mat_diff -= len(board.pieces(piece_type_val, chess.BLACK)) * val
                features["material_diff"] = mat_diff
                
                # You can add more features here...
                # features["fen_state"] = board.fen() # If you want the full FEN
                features["is_whites_turn"] = 1 if board.turn == chess.WHITE else 0 # 0 for white's turn
                
                features["is_in_check"] = 1 if board.is_check() else 0
                
                white_king_initial_square = chess.E1
                black_king_initial_square = chess.E8

                has_white_castled_kingside = (board.king(chess.WHITE) == chess.G1 and board.piece_at(chess.F1) and board.piece_at(chess.F1).piece_type == chess.ROOK)
                has_white_castled_queenside = (board.king(chess.WHITE) == chess.C1 and board.piece_at(chess.D1) and board.piece_at(chess.D1).piece_type == chess.ROOK)
                features["white_has_castled"] = 1 if (has_white_castled_kingside or has_white_castled_queenside) else 0

                # Similar logic for Black (king on g8/c8, rook on f8/d8)
                has_black_castled_kingside = (board.king(chess.BLACK) == chess.G8 and board.piece_at(chess.F8) and board.piece_at(chess.F8).piece_type == chess.ROOK)
                has_black_castled_queenside = (board.king(chess.BLACK) == chess.C8 and board.piece_at(chess.D8) and board.piece_at(chess.D8).piece_type == chess.ROOK)
                features["black_has_castled"] = 1 if (has_black_castled_kingside or has_black_castled_queenside) else 0
                features["bq_on_board"] = 1 if board.pieces(chess.QUEEN, chess.BLACK) else 0
                all_extracted_board_features.append(features)
    
    if not all_prefix_san_strings:
        print("No valid prefixes collected. Check game generation or target_ply_lengths.")
        return np.array([]), []

    # 3. Get SONAR Embeddings for all collected prefix strings
    # - Tokenize all strings in all_prefix_san_strings
    # - Pad them to a consistent length (max length found among prefixes, or model_max_seq_len_val)
    # - Batch them and feed to token2vec_pipeline.predict_from_token_ids()

    #print(f"\nTokenizing {len(all_prefix_san_strings)} prefixes...")
    tokenized_prefixes_list = []
    max_len_for_padding = 0
    valid_prefix_indices = [] # Keep track of indices of prefixes that are not too long

    for idx, prefix_text in enumerate(tqdm(all_prefix_san_strings, desc="Tokenizing Prefixes")):
        tokenized = tokenizer_encoder_func(prefix_text)
        if len(tokenized) > model_max_seq_len_val:
            # print(f"Warning: Prefix at original index {idx} too long ({len(tokenized)} > {model_max_seq_len_val}), skipping.")
            continue # Skip this prefix if it's too long after tokenization
        
        tokenized_tensor = torch.tensor(tokenized, dtype=torch.long, device=device_val)
        tokenized_prefixes_list.append(tokenized_tensor)
        valid_prefix_indices.append(idx) # Store index of the valid prefix
        if tokenized_tensor.shape[0] > max_len_for_padding:
            max_len_for_padding = tokenized_tensor.shape[0]
            
    if not tokenized_prefixes_list:
        print("No prefixes remaining after length filtering during tokenization.")
        return np.array([]), []

    # Filter all_extracted_board_features to match the valid (kept) prefixes
    filtered_board_features = [all_extracted_board_features[i] for i in valid_prefix_indices]
    # also filter all_prefix_san_strings if you plan to return it for inspection
    filtered_all_prefix_san_strings = [all_prefix_san_strings[i] for i in valid_prefix_indices]


    padded_prefix_tensors_list = []
    #print(f"Padding {len(tokenized_prefixes_list)} prefixes to max length {max_len_for_padding}...")
    for token_tensor in tqdm(tokenized_prefixes_list, desc="Padding Prefixes"):
        pad_amount = max_len_for_padding - token_tensor.shape[0]
        padded_tensor = torch.nn.functional.pad(token_tensor, (0, pad_amount), value=pad_idx_val)
        padded_prefix_tensors_list.append(padded_tensor)
    
    stacked_padded_prefix_tensors = torch.stack(padded_prefix_tensors_list)

    #print(f"Generating SONAR embeddings for {len(stacked_padded_prefix_tensors)} prefixes...")
    batch_embeddings_list_np = []
    INFERENCE_BATCH_SIZE = 128 # Or make this a parameter

    with torch.no_grad():
        for i in tqdm(range(0, len(stacked_padded_prefix_tensors), INFERENCE_BATCH_SIZE), desc="Embedding Prefixes"):
            batch_token_ids = stacked_padded_prefix_tensors[i:i+INFERENCE_BATCH_SIZE]
            # Assuming predict_from_token_ids returns sentence embeddings as the first element
            batch_sentence_embeddings, _, _, _ = token2vec_pipeline.predict_from_token_ids(batch_token_ids)
            batch_embeddings_list_np.append(batch_sentence_embeddings.cpu().numpy())
    
    if not batch_embeddings_list_np:
        print("Embedding generation resulted in an empty list.")
        return np.array([]), []
        
    final_prefix_embeddings_np = np.concatenate(batch_embeddings_list_np, axis=0)
    
    #print(f"Generated {final_prefix_embeddings_np.shape[0]} embeddings for {len(filtered_board_features)} feature sets.")
    assert final_prefix_embeddings_np.shape[0] == len(filtered_board_features), "Mismatch after embedding!"

    # Return the embeddings and the corresponding list of feature dictionaries
    return final_prefix_embeddings_np, filtered_board_features # Optionally return filtered_all_prefix_san_strings
#%%
global DEVICE, RANDOM_STATE, MODEL_NAME, OUTPUT_DIR
# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
MODEL_NAME = "text_sonar_basic_encoder"

OUTPUT_DIR = "../data/"

# --- Custom Text to Embedding Pipeline & Model Max Sequence Length ---
token2vec = CustomTextToEmbeddingPipeline(
    encoder=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=DEVICE,
)
    
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
#%%
# --- Example of how you might call this and then train a probe ---
# Assuming you have your SONAR pipeline components (token2vec, tokenizer_encoder, etc.)

model_max_seq_len = token2vec.model.encoder_frontend.pos_encoder.max_seq_len # Get this from your model
PAD_IDX = VOCAB_INFO.pad_idx # Get this from your tokenizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features = ['wq_on_board',
 'wk_on_g1',
 'e4_white_occupied',
 'material_diff',
 'is_whites_turn',
 'is_in_check',
 'white_has_castled',
 'black_has_castled',
 'bq_on_board']
for chosen_feature in features: 
        
    embeddings_for_probing, board_features_for_probing = generate_data_for_probing(
        num_games_to_process=100, # Number of games to sample prefixes from
        target_ply_lengths=[10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61],  # Create datasets for 5-move and 10-move prefixes
        token2vec_pipeline=token2vec,
        tokenizer_encoder_func=tokenizer_encoder,
        model_max_seq_len_val=model_max_seq_len,
        pad_idx_val=PAD_IDX,
        device_val=DEVICE,
        max_moves_in_full_game=60 # Ensure games are long enough for max ply
    )

    if embeddings_for_probing.shape[0] > 0:
        # Now you can train probes, e.g., to predict "wq_on_board"
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        if chosen_feature == 'material_diff':
            y_target_feature = np.array([f[chosen_feature] < 0 for f in board_features_for_probing], dtype=int)  # Convert to binary
        else:
            y_target_feature = np.array([f[chosen_feature] for f in board_features_for_probing])
        
        # Ensure there's more than one class for stratification and meaningful accuracy
        if len(np.unique(y_target_feature)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings_for_probing, y_target_feature, test_size=0.25, random_state=RANDOM_STATE, stratify=y_target_feature
            )
            probe_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
            probe_model.fit(X_train, y_train)
            predictions = probe_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            print(f"Accuracy of probe for '{chosen_feature}': {accuracy:.4f}")
            
            # Compare to a dummy classifier (predicting majority class)
            counts = np.bincount(y_test)
            majority_class_accuracy = np.max(counts) / len(y_test)
            print(f"Baseline (majority class) accuracy: {majority_class_accuracy:.4f}")

        else:
            print(f"Not enough class variance in '{chosen_feature}' to train a meaningful probe.")
    else:
        print("No data generated for probing.")
# %%
