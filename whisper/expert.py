from collections import OrderedDict
from typing import Dict, List, Union
from transformers import AutoFeatureExtractor, WhisperModel, WhisperForConditionalGeneration
import torch.nn as nn
from torch import Tensor
import torch
import dill
from ..interfaces import UpstreamBase
from scipy.io import wavfile

HIDDEN_DIM = 512
SAMPLE_RATE = 16000


class UpstreamExpert(nn.Module):
    def __init__(self, variant, mask:bool, ckpt: str = None, model_config: str = None, **kwargs):
        """
        Args:
            ckpt:
                The checkpoint path for loading your pretrained weights.
                Can be assigned by the -k option in run_downstream.py

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.
                Can be assigned by the -g option in run_downstream.py
        """
        super().__init__()
        self.name = "whisper"

        self.extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
        #if ckpt:
        #    self.model = WhisperModel.from_pretrained(ckpt, attn_implementation="eager")
        #else:
        #    self.model = WhisperModel.from_pretrained("openai/whisper-base", attn_implementation="eager")
        self.variant = variant
        self.mask = mask

        if variant == "pt":
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base", attn_implementation="eager")
            #self.model = WhisperModel.from_pretrained("openai/whisper-base", attn_implementation="eager")
        if variant == "ft":
            self.model = WhisperForConditionalGeneration.from_pretrained("/project/thesis/model/whisper-base-ft/checkpoint-21", attn_implementation="eager")

        self.decoder_input_ids = Tensor([[1, 1]]) * self.model.config.decoder_start_token_id
        self.state_dict_og = self.model.state_dict()


    def get_downsample_rates(self, key: str) -> int:
        """
        Since we do not do any downsampling in this example upstream
        All keys' corresponding representations have downsample rate of 1
        """
        return 160

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """
        device = wavs[0].device
        wavs = [wav.detach().cpu().numpy() for wav in wavs]
        input_values = self.extractor(wavs, do_normalize=True, sampling_rate=SAMPLE_RATE,return_tensors="pt").input_features
        input_values= input_values.to(device)

        if self.mask:
            #print("mask true")
            with open(f"/project/thesis/head_scores/mask_to_use.pkl", 'rb') as inp:  # use merged data
                mask = dill.load(inp)
                mask = mask.to(device)
                output_values = self.model(input_values, output_hidden_states=True, head_mask=mask,
                                           decoder_input_ids=self.decoder_input_ids.to(device).long())
        else:
            output_values = self.model(input_values,output_hidden_states=True, decoder_input_ids=self.decoder_input_ids.to(device).long())

        return {"hidden_states": output_values.encoder_hidden_states,
                "last_hidden_state": output_values.encoder_last_hidden_state }
