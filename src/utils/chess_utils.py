import random
import chess

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