import torch
from fairseq2.models.sequence import SequenceBatch
from sonar.inference_pipelines.text import (
    TextToTextModelPipeline,
)
from sonar.models.sonar_text import load_sonar_tokenizer

LANGUAGE_TOKEN_IDS = {
    "亵": 256000,
    "ace_Arab": 256001,
    "ace_Latn": 256002,
    "acm_Arab": 256003,
    "acq_Arab": 256004,
    "aeb_Arab": 256005,
    "afr_Latn": 256006,
    "ajp_Arab": 256007,
    "aka_Latn": 256008,
    "amh_Ethi": 256009,
    "apc_Arab": 256010,
    "arb_Arab": 256011,
    "ars_Arab": 256012,
    "ary_Arab": 256013,
    "arz_Arab": 256014,
    "asm_Beng": 256015,
    "ast_Latn": 256016,
    "awa_Deva": 256017,
    "ayr_Latn": 256018,
    "azb_Arab": 256019,
    "azj_Latn": 256020,
    "bak_Cyrl": 256021,
    "bam_Latn": 256022,
    "ban_Latn": 256023,
    "bel_Cyrl": 256024,
    "bem_Latn": 256025,
    "ben_Beng": 256026,
    "bho_Deva": 256027,
    "bjn_Arab": 256028,
    "bjn_Latn": 256029,
    "bod_Tibt": 256030,
    "bos_Latn": 256031,
    "bug_Latn": 256032,
    "bul_Cyrl": 256033,
    "cat_Latn": 256034,
    "ceb_Latn": 256035,
    "ces_Latn": 256036,
    "cjk_Latn": 256037,
    "ckb_Arab": 256038,
    "crh_Latn": 256039,
    "cym_Latn": 256040,
    "dan_Latn": 256041,
    "deu_Latn": 256042,
    "dik_Latn": 256043,
    "dyu_Latn": 256044,
    "dzo_Tibt": 256045,
    "ell_Grek": 256046,
    "eng_Latn": 256047,
    "epo_Latn": 256048,
    "est_Latn": 256049,
    "eus_Latn": 256050,
    "ewe_Latn": 256051,
    "fao_Latn": 256052,
    "pes_Arab": 256053,
    "fij_Latn": 256054,
    "fin_Latn": 256055,
    "fon_Latn": 256056,
    "fra_Latn": 256057,
    "fur_Latn": 256058,
    "fuv_Latn": 256059,
    "gla_Latn": 256060,
    "gle_Latn": 256061,
    "glg_Latn": 256062,
    "grn_Latn": 256063,
    "guj_Gujr": 256064,
    "hat_Latn": 256065,
    "hau_Latn": 256066,
    "heb_Hebr": 256067,
    "hin_Deva": 256068,
    "hne_Deva": 256069,
    "hrv_Latn": 256070,
    "hun_Latn": 256071,
    "hye_Armn": 256072,
    "ibo_Latn": 256073,
    "ilo_Latn": 256074,
    "ind_Latn": 256075,
    "isl_Latn": 256076,
    "ita_Latn": 256077,
    "jav_Latn": 256078,
    "jpn_Jpan": 256079,
    "kab_Latn": 256080,
    "kac_Latn": 256081,
    "kam_Latn": 256082,
    "kan_Knda": 256083,
    "kas_Arab": 256084,
    "kas_Deva": 256085,
    "kat_Geor": 256086,
    "knc_Arab": 256087,
    "knc_Latn": 256088,
    "kaz_Cyrl": 256089,
    "kbp_Latn": 256090,
    "kea_Latn": 256091,
    "khm_Khmr": 256092,
    "kik_Latn": 256093,
    "kin_Latn": 256094,
    "kir_Cyrl": 256095,
    "kmb_Latn": 256096,
    "kon_Latn": 256097,
    "kor_Hang": 256098,
    "kmr_Latn": 256099,
    "lao_Laoo": 256100,
    "lvs_Latn": 256101,
    "lij_Latn": 256102,
    "lim_Latn": 256103,
    "lin_Latn": 256104,
    "lit_Latn": 256105,
    "lmo_Latn": 256106,
    "ltg_Latn": 256107,
    "ltz_Latn": 256108,
    "lua_Latn": 256109,
    "lug_Latn": 256110,
    "luo_Latn": 256111,
    "lus_Latn": 256112,
    "mag_Deva": 256113,
    "mai_Deva": 256114,
    "mal_Mlym": 256115,
    "mar_Deva": 256116,
    "min_Latn": 256117,
    "mkd_Cyrl": 256118,
    "plt_Latn": 256119,
    "mlt_Latn": 256120,
    "mni_Beng": 256121,
    "khk_Cyrl": 256122,
    "mos_Latn": 256123,
    "mri_Latn": 256124,
    "zsm_Latn": 256125,
    "mya_Mymr": 256126,
    "nld_Latn": 256127,
    "nno_Latn": 256128,
    "nob_Latn": 256129,
    "npi_Deva": 256130,
    "nso_Latn": 256131,
    "nus_Latn": 256132,
    "nya_Latn": 256133,
    "oci_Latn": 256134,
    "gaz_Latn": 256135,
    "ory_Orya": 256136,
    "pag_Latn": 256137,
    "pan_Guru": 256138,
    "pap_Latn": 256139,
    "pol_Latn": 256140,
    "por_Latn": 256141,
    "prs_Arab": 256142,
    "pbt_Arab": 256143,
    "quy_Latn": 256144,
    "ron_Latn": 256145,
    "run_Latn": 256146,
    "rus_Cyrl": 256147,
    "sag_Latn": 256148,
    "san_Deva": 256149,
    "sat_Beng": 256150,
    "scn_Latn": 256151,
    "shn_Mymr": 256152,
    "sin_Sinh": 256153,
    "slk_Latn": 256154,
    "slv_Latn": 256155,
    "smo_Latn": 256156,
    "sna_Latn": 256157,
    "snd_Arab": 256158,
    "som_Latn": 256159,
    "sot_Latn": 256160,
    "spa_Latn": 256161,
    "als_Latn": 256162,
    "srd_Latn": 256163,
    "srp_Cyrl": 256164,
    "ssw_Latn": 256165,
    "sun_Latn": 256166,
    "swe_Latn": 256167,
    "swh_Latn": 256168,
    "szl_Latn": 256169,
    "tam_Taml": 256170,
    "tat_Cyrl": 256171,
    "tel_Telu": 256172,
    "tgk_Cyrl": 256173,
    "tgl_Latn": 256174,
    "tha_Thai": 256175,
    "tir_Ethi": 256176,
    "taq_Latn": 256177,
    "taq_Tfng": 256178,
    "tpi_Latn": 256179,
    "tsn_Latn": 256180,
    "tso_Latn": 256181,
    "tuk_Latn": 256182,
    "tum_Latn": 256183,
    "tur_Latn": 256184,
    "twi_Latn": 256185,
    "tzm_Tfng": 256186,
    "uig_Arab": 256187,
    "ukr_Cyrl": 256188,
    "umb_Latn": 256189,
    "urd_Arab": 256190,
    "uzn_Latn": 256191,
    "vec_Latn": 256192,
    "vie_Latn": 256193,
    "war_Latn": 256194,
    "wol_Latn": 256195,
    "xho_Latn": 256196,
    "ydd_Hebr": 256197,
    "yor_Latn": 256198,
    "yue_Hant": 256199,
    "zho_Hans": 256200,
    "zho_Hant": 256201,
    "zul_Latn": 256202,
}


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
        encoder_language: str = "eng_Latn",
        decoder_language: str = "eng_Latn",
    ):
        self.model_name_encoder = model_name_encoder
        self.model_name_decoder = model_name_decoder
        self.device = torch.device(device)

        # todo add support for other languages
        self._text_to_text_pipeline = None

        self.tokenizer = load_sonar_tokenizer(model_name_encoder)
        self.tokenizer_encoder = self.tokenizer.create_encoder()

        if (
            encoder_language not in LANGUAGE_TOKEN_IDS
            or decoder_language not in LANGUAGE_TOKEN_IDS
        ):
            raise ValueError(
                f"Language {encoder_language=} and {decoder_language=} must be one of {LANGUAGE_TOKEN_IDS.keys()}"
            )

        self.encoder_language_token_id = LANGUAGE_TOKEN_IDS[encoder_language]
        self.decoder_language_token_id = LANGUAGE_TOKEN_IDS[decoder_language]

    @property
    def text_to_text_pipeline(self) -> TextToTextModelPipeline:
        if self._text_to_text_pipeline is None:
            self._text_to_text_pipeline = TextToTextModelPipeline(
                encoder=self.model_name_encoder,
                decoder=self.model_name_decoder,
                tokenizer=self.model_name_encoder,
                device=self.device,
            )
        return self._text_to_text_pipeline

    @property
    def encoder(self):
        return self.text_to_text_pipeline.model.encoder

    @property
    def decoder(self):
        return self.text_to_text_pipeline.model.decoder

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
                    (n_rows, 1), self.encoder_language_token_id, device=self.device
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
                    self.decoder_language_token_id,  # second token must be the language token
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
