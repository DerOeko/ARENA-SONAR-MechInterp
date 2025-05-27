import torch
from fairseq2.models.sequence import SequenceBatch
from sonar.inference_pipelines.text import (
    TextToTextModelPipeline,
)
from sonar.models.sonar_text import load_sonar_tokenizer


class SonarEncoderDecoder:
    """
    Utility class for encoding and decoding text using the Sonar model.
    Only supports English.
    """

    def __init__(
        self,
        device: str,
        model_name_encoder: str = "text_sonar_basic_encoder",
        model_name_decoder: str = "text_sonar_basic_decoder",
    ):
        self.device = torch.device(device)

        # todo add support for other languages
        text_to_text_pipeline = TextToTextModelPipeline(
            encoder=model_name_encoder,
            decoder=model_name_decoder,
            tokenizer=model_name_encoder,
            device=self.device,
        )

        self.tokenizer = load_sonar_tokenizer(model_name_encoder)
        self.tokenizer_encoder = self.tokenizer.create_encoder()

        self.encoder = text_to_text_pipeline.model.encoder
        self.decoder = text_to_text_pipeline.model.decoder
        self.english_latin_token_id = 256047

    def get_vocab_id(self, token: str) -> int:
        """
        Get the token ID for a given token string.
        """
        tokens = self.tokenizer_encoder(token)
        if len(tokens) > 3:
            raise RuntimeError("Multiple tokens found for token: " + token)
        if len(tokens) == 2:
            raise RuntimeError("Empty token found for token: " + token)
        return tokens[1].item()

    def list_str_to_token_ids(self, tokens: list[str]) -> torch.Tensor:
        """
        Convert a list of token strings to a tensor of token IDs.
        """
        return torch.tensor(
            [self.get_vocab_id(token) for token in tokens], device=self.device
        )

    def list_str_to_token_ids_batch(
        self, token_id_lists: list[list[str]]
    ) -> torch.Tensor:
        """
        Convert a list of lists of token strings to a tensor of batchedtoken IDs.
        """
        assert len(set(len(token_id_list) for token_id_list in token_id_lists)) == 1, (
            "All token_id_lists must have the same length"
        )
        return torch.stack(
            [
                self.list_str_to_token_ids(token_id_list)
                for token_id_list in token_id_lists
            ],
            dim=0,
        )

    def token_ids_to_list_str(self, token_ids: list[int]) -> list[str]:
        """
        Convert a list of token IDs to a list of token strings.
        """
        return [self.tokenizer.model.index_to_token(token_id) for token_id in token_ids]

    def token_ids_to_list_str_batch(
        self, token_ids: list[list[int]]
    ) -> list[list[str]]:
        """
        Convert a list of lists of token IDs to a list of lists of token strings.
        """
        return [self.token_ids_to_list_str(token_id) for token_id in token_ids]

    def encode(
        self,
        token_ids: torch.Tensor,  # Annotated[torch.Tensor, "batch", "token_ids"]
    ) -> tuple[
        torch.Tensor,  # Annotated[torch.Tensor, "batch", "sequence_embedding"],
        torch.Tensor,  # Annotated[torch.Tensor, "batch", "token_ids", "embedding"],
    ]:
        """
        Encode a batch of token IDs into sentence embeddings (after pooling) and encoded sequences (before pooling).

        All input sequences must have the same length.
        """
        token_ids = token_ids.to(self.device)
        n_rows = token_ids.shape[0]
        token_ids = torch.cat(
            [
                torch.full(
                    (n_rows, 1), self.english_latin_token_id, device=self.device
                ),
                token_ids,
                torch.full(
                    (n_rows, 1), self.tokenizer.vocab_info.eos_idx, device=self.device
                ),
            ],
            dim=1,
        )
        encoder_output = self.encoder.forward(
            SequenceBatch(
                token_ids,
                None,
            )
        )
        return encoder_output.sentence_embeddings, encoder_output.encoded_seqs

    def decode_single(
        self,
        embeddings: torch.Tensor,
        max_length: int = 100,
    ) -> torch.Tensor:
        """
        Decode a single sentence embedding into a list of token IDs.
        """
        if len(embeddings.shape) == 2:
            # if sentence embedding without batch dimension then we need to add a dimension
            embeddings = embeddings.unsqueeze(1)

        seq = torch.tensor(
            [
                [
                    self.tokenizer.vocab_info.eos_idx,  # first token must be the end of sequence token for some reason
                    self.english_latin_token_id,  # second token must be the language token
                ]
            ],
            device=self.device,
        )

        # autoregressively generate the output sequence
        for _ in range(max_length):
            decoder_output = self.decoder.decode(
                seqs=seq,
                padding_mask=None,
                encoder_output=embeddings,
                encoder_padding_mask=None,
            )[0]
            # the decoder output is the output of layernorm, need to project to logits
            greedy_token = (
                self.decoder.project(decoder_output, decoder_padding_mask=None)
                .logits[:, -1, :]  # last embedding of latest token
                .argmax(dim=-1)  # take the most likely next token
            )
            seq = torch.cat(
                [seq, greedy_token.unsqueeze(0)], dim=1
            )  # add the new token to the sequence
            if (
                greedy_token[-1] == self.tokenizer.vocab_info.eos_idx
            ):  # if the new token is the end of sequence token, stop
                break
        return seq.squeeze(0).tolist()

    def decode(
        self,
        embeddings: torch.Tensor,
        max_length: int = 100,
    ) -> list[list[int]]:
        """
        Decode a batch of sentence embeddings into a list of lists of token IDs.
        """
        return [
            self.decode_single(embeddings[i, ...].unsqueeze(0), max_length)
            for i in range(embeddings.shape[0])
        ]
