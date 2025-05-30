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

