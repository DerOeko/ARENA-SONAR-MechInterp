import random
import chess
from tqdm.auto import tqdm
import numpy as np
import torch
BOARD_POSITION_LETTERS = "abcdefgh"
BOARD_POSITION_NUMBERS = "12345678"
BOARD_POSITIONS = [
    f"{letter}{number}"
    for letter in BOARD_POSITION_LETTERS
    for number in BOARD_POSITION_NUMBERS
]

def make_move_illegal(move: str) -> str:
    """
    Make a move illegal by randomising the destination position.
    """
    final_char = ""
    if move[-1] not in BOARD_POSITION_NUMBERS:
        final_char = move[-1]
        move = move[:-1]

    if len(move) < 2 or (
        move[-2] not in BOARD_POSITION_LETTERS
        and move[-1] not in BOARD_POSITION_NUMBERS
    ):
        # If the move isn't defining a destination position, don't change it
        return move + final_char
    return move[:-2] + random.choice(BOARD_POSITIONS) + final_char


def make_sequence_illegal(move_steps: list[str]) -> list[str]:
    """
    Make a sequence illegal by randomising the destination position of each move.

    Parameters
    ----------
    move_steps : str
        A string of moves separated by spaces.
        Each game is separated by a newline, single line for a single game.

    Returns
    -------
    str
        A string of moves with the destination position randomised.
    """
    game_strings = []
    for game in move_steps:
        game_strings.append(" ".join(map(make_move_illegal, game.split(" "))))
    return game_strings

def generate_legal_chess_sequences(num_sequences: int, max_moves_per_sequence: int, use_san: bool = False) -> list[str]:
    """
    Generates a list of legal chess game sequences.

    Args:
        num_sequences (int): The number of game sequences to generate.
        max_moves_per_sequence (int): The maximum number of moves for each sequence.
        use_san (bool): If True, use Standard Algebraic Notation (SAN), otherwise UCI.

    Returns:
        list[str]: A list where each item is a space-separated string of moves
                   representing a game sequence.
    """
    generated_sequences = []
    for _ in range(num_sequences):
        board = chess.Board()
        current_sequence_moves = []

        for _ in range(max_moves_per_sequence):
            legal_moves = list(board.legal_moves)
            if not legal_moves or board.is_game_over():
                break

            move = random.choice(legal_moves)
            
            if use_san:
                current_sequence_moves.append(board.san(move))
            else:
                current_sequence_moves.append(move.uci()) # .uci() is more explicit than str()
            
            board.push(move)
        
        if current_sequence_moves:
            generated_sequences.append(" ".join(current_sequence_moves))
            
    return generated_sequences



def make_one_san_move_illegal_by_random_destination(move_san: str) -> str:
    """
    Makes a single SAN move "illegal" by randomizing its destination square.
    Handles common suffixes like '+' or '#'.
    NOTE: Will likely produce syntactically incorrect SAN for castling and promotion.
    """
    if not move_san: # Handle empty string case
        return random.choice(BOARD_POSITIONS)

    original_move = move_san
    suffix = ""

    # Check for common suffixes (check, checkmate)
    if original_move.endswith("#") or original_move.endswith("+"):
        suffix = original_move[-1]
        original_move = original_move[:-1]
    
    # If after stripping suffix, the move is too short, or it's special (castling/promotion)
    # this simple logic will struggle. For now, we apply the core logic.
    # Castling and promotion will likely become syntactically invalid.
    # Example: "O-O" -> final_char="O", move="O-", move[:-2]="", returns random_square + "O" -> "a1O" (bad)
    # Example: "e8=Q" -> final_char="Q", move="e8=", move[:-2]="e", returns "e" + random_square + "Q" -> "ea1Q" (bad)

    prefix = ""
    if len(original_move) >= 2: # Need at least two chars for a destination
        prefix = original_move[:-2]
    elif len(original_move) == 1: # e.g. a piece letter if original was very short like "N+"
        prefix = original_move[0] 


    # For very short moves like "e4", prefix will be ""
    # For "Nf3", prefix will be "N"
    # For "Raxe4", prefix will be "Rax"

    return prefix + random.choice(BOARD_POSITIONS) + suffix

def generate_illegal_game_sequences(legal_sequences, num_corruptions_per_sequence=1):
    illegal_game_sequences = []
    for seq_str in legal_sequences:
        original_moves = seq_str.split(" ")
        
        if not original_moves: # Handle case of an empty sequence string
            # illegal_game_sequences.append("") # Or skip
            continue

        num_total_moves_in_seq = len(original_moves)
        # This will be the list of moves we modify
        working_moves_list = list(original_moves) 

        # Identify suitable (non-special) indices and special indices
        non_special_indices = []
        special_indices = []
        for i, move_san in enumerate(working_moves_list):
            is_castling = (move_san == "O-O" or move_san == "O-O-O")
            is_promotion = ('=' in move_san)
            if not is_castling and not is_promotion:
                non_special_indices.append(i)
            else:
                special_indices.append(i)
        
        # Shuffle them to pick randomly if we have more candidates than needed
        random.shuffle(non_special_indices)
        random.shuffle(special_indices)

        # Determine the actual number of corruptions we can make (up to num_corruptions_per_sequence)
        # and collect the distinct indices to corrupt
        indices_actually_corrupted_this_sequence = set()

        # Corrupt non-special moves first
        for idx in non_special_indices:
            if len(indices_actually_corrupted_this_sequence) < num_corruptions_per_sequence:
                working_moves_list[idx] = make_one_san_move_illegal_by_random_destination(working_moves_list[idx])
                indices_actually_corrupted_this_sequence.add(idx)
            else:
                break # Reached desired number of corruptions
                
        # If more corruptions are needed and we haven't reached the target, corrupt special moves
        if len(indices_actually_corrupted_this_sequence) < num_corruptions_per_sequence:
            for idx in special_indices:
                if idx not in indices_actually_corrupted_this_sequence: # Ensure it wasn't somehow already picked (shouldn't happen here)
                    if len(indices_actually_corrupted_this_sequence) < num_corruptions_per_sequence:
                        working_moves_list[idx] = make_one_san_move_illegal_by_random_destination(working_moves_list[idx])
                        indices_actually_corrupted_this_sequence.add(idx)
                    else:
                        break # Reached desired number of corruptions
        
        # The working_moves_list now contains the accumulated corruptions
        illegal_seq_str = " ".join(working_moves_list)
        illegal_game_sequences.append(illegal_seq_str)

    # --- Display some results ---
    print(f"Target corruptions per sequence: {num_corruptions_per_sequence}")
    print(f"Processed {len(legal_sequences)} legal sequences.")
    print(f"Generated {len(illegal_game_sequences)} illegal sequences.")

    for i in range(min(5, len(legal_sequences))):
        print("-" * 20)
        print(f"Original Legal Seq {i+1}: {legal_sequences[i]}")
        if i < len(illegal_game_sequences):
            # To see which moves changed, you could compare them one by one
            original_split = legal_sequences[i].split()
            illegal_split = illegal_game_sequences[i].split()
            changed_indices = [k for k in range(len(original_split)) if original_split[k] != illegal_split[k]]
            print(f"Generated Illegal Seq {i+1}: {illegal_game_sequences[i]} (Corrupted at original indices: {changed_indices})")
            
    return illegal_game_sequences

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

    print(f"Generating game data for probing at ply lengths: {target_ply_lengths}...")

    # 1. Generate Legal Game Sequences (long enough to get desired prefixes)
    # We need full games to extract prefixes from.
    # Make sure max_moves_in_full_game is at least max(target_ply_lengths)
    if not target_ply_lengths:
        print("Warning: No target_ply_lengths specified for prefixes.")
        return np.array([]), []
        
    actual_max_ply = max(target_ply_lengths)
    if max_moves_in_full_game < actual_max_ply // 2 + 1 : # ensure enough full moves
        max_moves_in_full_game = actual_max_ply // 2 + 5


    print(f"Generating {num_games_to_process} base games up to {max_moves_in_full_game} moves each...")
    base_legal_games_san = generate_legal_chess_sequences(
        num_sequences=num_games_to_process,
        max_moves_per_sequence=max_moves_in_full_game, # Generate reasonably long games
        use_san=True
    )

    # 2. For each game, extract prefixes and their board states
    print(f"Extracting prefixes and board features...")
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

                all_extracted_board_features.append(features)
    
    if not all_prefix_san_strings:
        print("No valid prefixes collected. Check game generation or target_ply_lengths.")
        return np.array([]), []

    # 3. Get SONAR Embeddings for all collected prefix strings
    # This part will be very similar to your existing embedding generation pipeline:
    # - Tokenize all strings in all_prefix_san_strings
    # - Pad them to a consistent length (max length found among prefixes, or model_max_seq_len_val)
    # - Batch them and feed to token2vec_pipeline.predict_from_token_ids()

    print(f"\nTokenizing {len(all_prefix_san_strings)} prefixes...")
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
    print(f"Padding {len(tokenized_prefixes_list)} prefixes to max length {max_len_for_padding}...")
    for token_tensor in tqdm(tokenized_prefixes_list, desc="Padding Prefixes"):
        pad_amount = max_len_for_padding - token_tensor.shape[0]
        padded_tensor = torch.nn.functional.pad(token_tensor, (0, pad_amount), value=pad_idx_val)
        padded_prefix_tensors_list.append(padded_tensor)
    
    stacked_padded_prefix_tensors = torch.stack(padded_prefix_tensors_list)

    print(f"Generating SONAR embeddings for {len(stacked_padded_prefix_tensors)} prefixes...")
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
    
    print(f"Generated {final_prefix_embeddings_np.shape[0]} embeddings for {len(filtered_board_features)} feature sets.")
    assert final_prefix_embeddings_np.shape[0] == len(filtered_board_features), "Mismatch after embedding!"

    # Return the embeddings and the corresponding list of feature dictionaries
    return final_prefix_embeddings_np, filtered_board_features # Optionally return filtered_all_prefix_san_strings