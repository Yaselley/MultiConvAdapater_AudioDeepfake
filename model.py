import torch
import torch.nn as nn
import torch.nn.functional as F
from aasist import AASIST
from wav2vec2 import Wav2Vec2Model, Wav2Vec2Config
from config import W2V_XLSR_CONFIG, W2V_XLSR_CKPT
import json 

class SSLModel(nn.Module):
    def __init__(self, 
                 config,
                 ckpt_path=W2V_XLSR_CKPT,
                 config_path=W2V_XLSR_CONFIG,
                 training=True):
        """
        SSLModel can be initialized in two modes:
          - training=True: requires ckpt_path to load weights and save config.
          - training=False: loads model from saved config only (no checkpoint needed).
        """
        super().__init__()

        if training:
            assert ckpt_path is not None, "ckpt_path must be provided in training mode."
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ckpt_config = ckpt["cfg"]["model"]

            model_cfg = Wav2Vec2Config(**ckpt_config)
            model_cfg.kernel_sizes = config["kernel_sizes"]

            self.w2v = Wav2Vec2Model(model_cfg)

            # Load weights
            missing_keys, unexpected_keys = self.w2v.load_state_dict(ckpt["model"], strict=False)

            # Freeze all params
            for param in self.w2v.parameters():
                param.requires_grad = False

            # Optionally unfreeze missing keys
            named_params = dict(self.w2v.named_parameters())
            for name in missing_keys:
                if name in named_params:
                    named_params[name].requires_grad = True

        else:
            # Inference mode: only use config (no weights)
            with open(config_path, "r") as f:
                cfg_dict = json.load(f)

            model_cfg = Wav2Vec2Config(**cfg_dict)
            model_cfg.kernel_sizes = config["kernel_sizes"]

            self.w2v = Wav2Vec2Model(model_cfg)

        # Store mode for clarity
        self.training_mode = training

    def forward(self, input_data):
        input_data = input_data.squeeze(1)
        x = self.w2v(input_data, mask=False, features_only=True)["x"]
        return x

    
class SSL_AASIST_Model(nn.Module):
    def __init__(self, config):
                
        super(SSL_AASIST_Model, self).__init__()

        self.ssl = SSLModel(config, training=config.get("training", True))
        self.aasist = AASIST()

    def forward(self, x):

        x = self.ssl(x)
        x = self.aasist(x)

        return x

    def freeze(self,):
        self.ssl.freeze_model()
