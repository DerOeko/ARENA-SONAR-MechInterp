# %%
import os
import random
from typing import Annotated, List

import numpy as np
import torch
from fairseq2.generation import BeamSearchSeq2SeqGenerator
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask
from fairseq2.typing import CPU, DataType, Device
from sonar.inference_pipelines.text import (
    EmbeddingToTextModelPipeline,
    TextToEmbeddingModelPipeline,
    TextToTextModelPipeline,
)
from sonar.models.sonar_text import load_sonar_tokenizer
from tqdm import tqdm


class SonarEncoderDecoder:
    def __init__(
        self,
        device: str,
        model_name_encoder: str = "text_sonar_basic_encoder",
        model_name_decoder: str = "text_sonar_basic_decoder",
    ):
        # todo add support for other languages
        # todo do I need to do something with padding masks?
        text_to_text_pipeline = TextToTextModelPipeline(
            encoder=model_name_encoder,
            decoder=model_name_decoder,
            tokenizer=model_name_encoder,
            device=device,
        )

        self.device = device
        self.tokenizer = load_sonar_tokenizer(model_name_encoder)
        # VOCAB_INFO = tokenizer.vocab_info
        # PAD_IDX = VOCAB_INFO.pad_idx
        # EOS_IDX = VOCAB_INFO.eos_idx
        # UNK_IDX = VOCAB_INFO.unk_idx
        # BOS_IDX = VOCAB_INFO.bos_idx

        self.encoder = text_to_text_pipeline.model.encoder
        self.decoder = text_to_text_pipeline.model.decoder

    def encode(
        self, token_ids: Annotated[torch.Tensor, "batch", "token_ids"]
    ) -> tuple[
        Annotated[torch.Tensor, "batch", "sequence_embedding"],
        Annotated[torch.Tensor, "batch", "token_ids", "embedding"],
    ]:
        token_ids = token_ids.to(self.device)
        token_ids = token_ids.concat(
            [
                torch.tensor(
                    [[self.tokenizer.vocab_info.ENGLISH_TOKEN_ID]], device=self.device
                ),
                token_ids,
                torch.tensor([[self.tokenizer.vocab_info.eos_idx]], device=self.device),
            ]
        )
        encoder_output = self.encoder.forward(
            SequenceBatch(
                token_ids,
                None,
            )
        )
        return encoder_output.sentence_embeddings, encoder_output.encoded_seqs

    def decode(
        self,
        embeddings: Annotated[torch.Tensor, "batch", "sequence_embedding"]
        | Annotated[torch.Tensor, "batch", "token_ids", "embedding"],
        max_length: int = 100,
    ) -> Annotated[torch.Tensor, "batch", "token_ids"]:
        if len(embeddings.shape) == 2:
            # if sentence embedding then we need to add a dimension
            embeddings = embeddings.unsqueeze(1)

        seqs = torch.tensor(
            [
                [
                    self.tokenizer.vocab_info.eos_idx,
                    self.tokenizer.vocab_info.ENGLISH_TOKEN_ID,
                ]
            ],
            device=self.device,
        )

        # todo this only works for one sequence at a time, fix this
        for i in range(max_length):
            decoder_output = self.decoder.decode(
                seqs=seqs,
                padding_mask=None,
                encoder_output=embeddings,
                encoder_padding_mask=None,
            )[0]
            greedy_token = (
                self.decoder.project(decoder_output, decoder_padding_mask=None)
                .logits[:, -1, :]
                .argmax(dim=-1)
            )
            seqs = torch.cat([seqs, greedy_token.unsqueeze(0)], dim=1)
            if greedy_token[-1] == self.tokenizer.vocab_info.eos_idx:
                break
        return seqs


# # --- Configuration ---
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# RANDOM_STATE = 42
# MODEL_NAME_ENCODER = "text_sonar_basic_encoder"
# MODEL_NAME_DECODER = "text_sonar_basic_decoder"

# # Seed for reproducibility
# random.seed(RANDOM_STATE)
# np.random.seed(RANDOM_STATE)
# torch.manual_seed(RANDOM_STATE)
# if DEVICE.type == "cuda":
#     torch.cuda.manual_seed_all(RANDOM_STATE)

# # %% --- Initialize Tokenizer and Model ---
# print("--- Initializing Tokenizer and Model ---")

# # Load tokenizer for special tokens
# tokenizer = load_sonar_tokenizer(MODEL_NAME_ENCODER)
# tokenizer_encoder = tokenizer.create_encoder()
# tokenizer_decoder = tokenizer.create_decoder()

# # Get special token IDs
# VOCAB_INFO = tokenizer.vocab_info
# PAD_IDX = VOCAB_INFO.pad_idx
# EOS_IDX = VOCAB_INFO.eos_idx
# UNK_IDX = VOCAB_INFO.unk_idx
# BOS_IDX = VOCAB_INFO.bos_idx

# # Get English language token ID
# dummy_tokenized = tokenizer_encoder("test")
# ENG_LANG_TOKEN_IDX = dummy_tokenized[0].item()

# # Load text-to-text model pipeline
# text_to_text_pipeline = TextToTextModelPipeline(
#     encoder=MODEL_NAME_ENCODER,
#     decoder=MODEL_NAME_DECODER,
#     tokenizer=MODEL_NAME_ENCODER,
#     device=DEVICE
# )


# print("Models initialized successfully.")

# # %%

# encoder = text_to_text_pipeline.model.encoder
# decoder = text_to_text_pipeline.model.decoder

# # %%

# def get_vocab_id(token: str) -> int:
#     tokens = tokenizer_encoder(token)
#     if len(tokens) > 3:
#         raise RuntimeError("Multiple tokens found for token: " + token)
#     if len(tokens) == 2:
#         raise RuntimeError("Empty token found for token: " + token)
#     return tokens[1].item()


# # %%
# house_idx = get_vocab_id("house")
# # %%

# # %% --- Test with a simple example ---
# encoder_output = encoder.forward(SequenceBatch(
#     torch.tensor([[
#         ENG_LANG_TOKEN_IDX,
#         get_vocab_id("I"),
#         get_vocab_id("am"),
#         get_vocab_id("a"),
#         get_vocab_id("happy"),
#         get_vocab_id("dog"),
#         EOS_IDX,
#     ]]).to(DEVICE), None)
# )
# encoder_output


# # Create initial token sequence with start token
# start_tokens = torch.tensor([[EOS_IDX, ENG_LANG_TOKEN_IDX]]).to(DEVICE)

# decoder_output = decoder.decode(
#     seqs=start_tokens,
#     padding_mask=None,
#     encoder_output=encoder_output.sentence_embeddings.unsqueeze(1),
#     # encoder_output=encoder_output.encoded_seqs,
#     encoder_padding_mask=None
# )
# decoder_output


# # %%
# greedy_token = decoder.project(decoder_output[0], decoder_padding_mask=None).logits.argmax()
# greedy_token
# # %%
# def get_token_from_id(vocab_id: int) -> str:
#     # The tokenizer's model has an index_to_token method
#     return text_to_text_pipeline.tokenizer.model.index_to_token(vocab_id)

# # Test it out with the house_idx
# token = get_token_from_id(house_idx)
# print(f"Token for index {house_idx}: {token}")

# # %%
# get_token_from_id(greedy_token)
# # %%

# seqs = torch.tensor([[EOS_IDX, ENG_LANG_TOKEN_IDX]]).to(DEVICE)

# for i in range(10):
#     decoder_output = decoder.decode(
#         seqs=seqs,
#         padding_mask=None,
#         # encoder_output=encoder_output.encoded_seqs,
#         encoder_output=encoder_output.sentence_embeddings.unsqueeze(1),
#         encoder_padding_mask=None
#     )
#     greedy_token = decoder.project(decoder_output[0], decoder_padding_mask=None).logits[:, -1, :].argmax(dim=-1)
#     seqs = torch.cat([seqs, greedy_token.unsqueeze(0)], dim=1)
#     print(seqs)
#     print([get_token_from_id(token_id) for token_id in seqs[0]])
#     if greedy_token[-1] == EOS_IDX:
#         break
# # %%
