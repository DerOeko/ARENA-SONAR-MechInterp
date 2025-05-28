import torch

def generate_random_sequences(
    num_sequences: int,
    vocab_size: int,
    seq_length: int,
    end_token_id: int,
    device: torch.device
) -> torch.Tensor:
    """
    Generates a batch of random token sequences, each of `seq_length`,
    with the last token being `end_token_id`.

    Args:
        num_sequences: Number of random sequences to generate.
        seq_length: The total length of each sequence (including the end_token_id).
        vocab_size: The size of the vocabulary to sample from.
        end_token_id: The token ID to place at the end of each sequence.
        device: The torch device to create tensors on.

    Returns:
        A Tensor of shape [num_sequences, seq_length] with random token IDs.
    """    
    if seq_length == 1:
        # If sequence length is 1, it's just the end token
        sequences = torch.full((num_sequences, 1), fill_value=end_token_id, dtype=torch.long, device=device)
    else:
        # Generate (seq_length - 1) random tokens for each sequence
        random_parts = torch.randint(0, vocab_size, (num_sequences, seq_length - 1), device=device)
        
        # Create a tensor for the end_token_id, repeated for each sequence
        end_tokens = torch.full((num_sequences, 1), fill_value=end_token_id, dtype=torch.long, device=device)
        
        # Concatenate the random parts with the end_token_id
        sequences = torch.cat((random_parts, end_tokens), dim=1)
        
    return sequences