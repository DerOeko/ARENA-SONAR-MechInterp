import torch
from typing import Optional, Tuple

from fairseq2.typing import Device, CPU, DataType # DataType is used in __init__
from fairseq2.nn.padding import PaddingMask
# from fairseq2.models.sequence import SequenceModelOutput # Optional: for type hinting intermediate outputs if you choose
                                                        # Your current version unpacks tuples directly.

# Base class for inheritance
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

class CustomTextToEmbeddingPipeline(TextToEmbeddingModelPipeline):
    def __init__(self, encoder, 
                 tokenizer, # TextTokenizer from fairseq2.data.text
                 device: Device = CPU, 
                 dtype: Optional[DataType] = None): # Ensure DataType is imported for the hint
        super().__init__(encoder, tokenizer, device, dtype)
        # self.model is an instance of SonarEncoderModel (specifically SonarTextTransformerEncoderModel)
        # self.tokenizer is TextTokenizer

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
        #    Using self.tokenizer.vocab_info (which is good, no external global needed here)
        padding_mask_bool_tensor = (token_ids_on_device != self.tokenizer.vocab_info.pad_idx)
        seq_lengths = padding_mask_bool_tensor.sum(dim=1)
        current_padding_mask = PaddingMask(seq_lengths, batch_seq_len=token_ids_on_device.shape[1])

        # --- Replicating SonarTextTransformerEncoderModel.forward() manually ---

        # 2. Encoder Frontend call
        initial_embeds_tensor, pm_after_frontend = self.model.encoder_frontend(
            token_ids_on_device, current_padding_mask
        )

        # 3. Encoder call
        encoded_seqs_tensor, pm_after_encoder = self.model.encoder(
            initial_embeds_tensor, pm_after_frontend
        )

        current_last_hidden_states = encoded_seqs_tensor 

        # 4. Optional LayerNorm
        if hasattr(self.model, 'layer_norm') and self.model.layer_norm is not None: # Check if layer_norm exists
            current_last_hidden_states = self.model.layer_norm(current_last_hidden_states)
        
        if steering_vector is not None:
            current_last_hidden_states = current_last_hidden_states.clone()

        # 5. Apply Steering
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
            steered_tokens = selected_tokens + steering_vector_dev
            current_last_hidden_states[batch_indices_for_steering, target_indices_dev] = steered_tokens
        
        # 6. Pool 
        sentence_embeddings = self.model.pool(
            current_last_hidden_states, pm_after_frontend, self.model.pooling # self.model.pooling refers to the enum
        )
        
        final_target_device = target_device or self.device
        padding_mask_to_return = pm_after_frontend 
        
        return (
            sentence_embeddings.to(final_target_device),
            current_last_hidden_states.to(final_target_device), 
            initial_embeds_tensor.to(final_target_device),
            padding_mask_to_return 
        )