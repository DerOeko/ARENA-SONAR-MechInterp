import random

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

    return move[:-2] + random.choice(BOARD_POSITIONS) + final_char


def make_sequence_illegal(move_steps: str) -> list[str]:
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
    for game in move_steps.split("\n"):
        game_strings.append(" ".join(map(make_move_illegal, move_steps.split(" "))))
    return "\n".join(game_strings)
