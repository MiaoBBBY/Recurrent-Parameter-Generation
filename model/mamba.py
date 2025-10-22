import torch
from torch import nn
from mamba_ssm import Mamba2 as Mamba
import math


class MambaModel(nn.Module):
    config = {}

    def __init__(self, positional_embedding):
        super().__init__()
        mamba_config = {
            "d_model": self.config["d_model"],
            "d_state": self.config["d_state"],
            "d_conv": self.config["d_conv"],
            "expand": self.config["expand"],
        }
        self.mamba_forward = nn.Sequential(*[Mamba(**mamba_config) for _ in range(self.config["num_layers"])])
        pe = positional_embedding[None, :, :]
        if self.config.get("trainable_pe"):
            self.pe = nn.Parameter(pe)
        else:  # fixed positional embedding
            self.register_buffer("pe", pe)

# in model/mamba.py -> class MambaModel

    def forward(self, output_shape, condition=None):
        # ... (我们之前添加的 print 语句可以保留或删除) ...
        
        # --- 在这里加入这三行调试代码 ---
        # print(f"[DEBUG MambaModel] self.pe device: {self.pe.device}")
        # print(f"[DEBUG MambaModel] Input condition device: {condition.device}")
        
        device = next(self.mamba_forward.parameters()).device
        pe_on_device = self.pe.to(device)
        condition_on_device = condition.to(device)
        combined_input = pe_on_device.repeat(output_shape[0], 1, 1) + condition_on_device

        # print(f"[DEBUG MambaModel] Final 'combined_input' device before mamba_forward: {combined_input.device}")
        
        x = self.mamba_forward(combined_input)
        return x