import random

BOARD_POSITIONS = [
    f"{letter}{number}" for letter in "abcdefgh" for number in "12345678"
]


def make_sequence_illegal(move_steps: list[str]) -> list[str]:
    """
    Make a sequence illegal by adding a random number of illegal tokens to the end of the sequence.
    """
    return [move[:-2] + random.choice(BOARD_POSITIONS) for move in move_steps]
