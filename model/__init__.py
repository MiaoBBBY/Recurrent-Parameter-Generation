import torch
from abc import ABC
from torch import nn
from torch.nn import functional as F
from .diffusion import DiffusionLoss, DDIMSampler, DDPMSampler
from .transformer import TransformerModel
from .mamba import MambaModel
from .lstm import LstmModel
from .gatemlp import GMLPModel


import torch
from abc import ABC
from torch import nn
from torch.nn import functional as F
from .diffusion import DiffusionLoss, DDIMSampler, DDPMSampler
from .transformer import TransformerModel
from .mamba import MambaModel
from .lstm import LstmModel
from .gatemlp import GMLPModel

# --- 1. 新增：通用的实例化工具函数 ---
import importlib

def instantiate_from_config(config):
    """根据配置动态实例化一个类"""
    if not "target" in config:
        raise KeyError("Expected key 'target' to instantiate.")
    module_path, class_name = config["target"].rsplit(".", 1)
    module = importlib.import_module(module_path, package=__name__)
    class_ = getattr(module, class_name)
    return class_(**config.get("params", dict()))


# --- 2. 新增：适配 Mamba 的 EmbedData 条件编码器 ---
# (为了方便，我们暂时将这个类放在这里，您可以之后将其移动到自己的文件中)
from .set_transformer.models import SetTransformer

class EmbedData(nn.Module):
    def __init__(self, enconfig, deconfig, d_model, ckpt_path=None, **kwargs):
        super(EmbedData, self).__init__()
        self.ckpt_path = ckpt_path
        self.d_model = d_model

        # 1. intra 使用原始的 enconfig, 它的输入是 768 维的 CLIP 特征
        self.intra = SetTransformer(enconfig, deconfig)

        # 2. 为 inter 创建一个新的 enconfig
        #    它的输入维度必须是 intra 的输出维度, 即 deconfig['dim_output'] (512)
        inter_enconfig = enconfig.copy() # 复制一份，避免修改原始字典
        inter_enconfig['dim_input'] = deconfig['dim_output'] 
        
        # 3. inter 使用新的、正确的 inter_enconfig
        self.inter = SetTransformer(inter_enconfig, deconfig)
        
        self.proj = nn.Linear(512, self.d_model)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
            for param in self.intra.parameters():
                param.requires_grad = False
            for param in self.inter.parameters():
                param.requires_grad = False

    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu",weights_only=True)
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        
        # 仅加载与当前模型匹配的键
        model_sd = self.state_dict()
        filtered_sd = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        
        self.load_state_dict(filtered_sd, strict=False)
        print(f"Restored EmbedData from {path}")

    def forward(self, inputs, sequence_length):
        outputs = []
        for x in inputs:
            if isinstance(x, list) and len(x) == 1:
                x = x[0]

            x = x.to(next(self.parameters()).device)
            z = self.intra(x).squeeze(1)
            z = z.unsqueeze(0)
            out = self.inter(z).reshape(-1)
            outputs.append(out)
        
        outputs = torch.stack(outputs, 0).reshape(-1, 512)
        
        # 投影到 d_model 维度
        c = self.proj(outputs) # -> 输出形状 (batch_size, d_model)

        # 通过 unsqueeze 和 repeat 适配 Mamba 的输入形状
        c = c.unsqueeze(1) # -> (batch_size, 1, d_model)
        c = c.repeat(1, sequence_length, 1) # -> (batch_size, sequence_length, d_model)
        
        return c

# ------------------------------------------------------------
# ------------------------------------------------------------  
# ------------------------------------------------------------          


class ModelDiffusion(nn.Module, ABC):
    config = {}

    def __init__(self, sequence_length):
        super().__init__()
        DiffusionLoss.config = self.config
        self.criteria = DiffusionLoss()
        if self.config.get("post_d_model") is None:
            assert self.config["d_model"] == self.config["condition_dim"]
        self.sequence_length = sequence_length
        # to define model after this function
        self.to_condition = nn.Linear(self.config["d_condition"], self.config["d_model"])
        self.to_permutation_state = nn.Embedding(self.config["num_permutation"], self.config["d_model"])
        self.to_permutation_state.weight = \
                nn.Parameter(torch.ones_like(self.to_permutation_state.weight) / self.config["d_model"])

    def forward(self, output_shape=None, x_0=None, condition=None, permutation_state=None, **kwargs):
        # condition
        if condition is not None:
            assert len(condition.shape) == 2
            assert condition.shape[-1] == self.config["d_condition"]
            condition = self.to_condition(condition.to(self.device)[:, None, :])
        else:  # not use condition
            condition = self.to_condition(torch.zeros(size=(1, 1, 1), device=self.device))
        # process
        if kwargs.get("sample"):
            if permutation_state is not False:
                permutation_state = torch.randint(0, self.to_permutation_state.num_embeddings, (1,), device=self.device)
                permutation_state = self.to_permutation_state(permutation_state)[:, None, :]
            else:  # permutation state == False
                permutation_state = 0.
            return self.sample(x=None, condition=condition+permutation_state)
        else:  # train
            if permutation_state is not None:
                permutation_state = self.to_permutation_state(permutation_state)[:, None, :]
            else:  # not use permutation state
                permutation_state = 0.
            # Given condition c and ground truth token x, compute loss
            c = self.model(output_shape, condition+permutation_state)
            loss = self.criteria(x=x_0, c=c, **kwargs)
            return loss

    @torch.no_grad()
    def sample(self, x=None, condition=None):
        z = self.model([1, self.sequence_length, self.config["d_model"]], condition)
        if x is None:
            x = torch.randn((1, self.sequence_length, self.config["model_dim"]), device=z.device)
        x = self.criteria.sample(x, z)
        return x

    @property
    def device(self):
        return next(self.parameters()).device


class ModelMSELoss(nn.Module, ABC):
    config = {}

    def __init__(self, sequence_length):
        super().__init__()
        if self.config.get("post_d_model") is None:
            assert self.config["d_model"] == self.config["condition_dim"]
        self.sequence_length = sequence_length
        # to define model after this function
        self.to_condition = nn.Linear(self.config["d_condition"], self.config["d_model"])
        self.to_permutation_state = nn.Embedding(self.config["num_permutation"], self.config["d_model"])
        self.to_permutation_state.weight = \
                nn.Parameter(torch.ones_like(self.to_permutation_state.weight) / self.config["d_model"])

    def forward(self, output_shape=None, x_0=None, condition=None, permutation_state=None, **kwargs):
        # condition
        if condition is not None:
            assert len(condition.shape) == 2
            assert condition.shape[-1] == self.config["d_condition"]
            condition = self.to_condition(condition.to(self.device)[:, None, :])
        else:  # not use condition
            condition = self.to_condition(torch.zeros(size=(1, 1, 1), device=self.device))
        # process
        if kwargs.get("sample"):
            if permutation_state is not False:
                permutation_state = torch.randint(0, self.to_permutation_state.num_embeddings, (1,), device=self.device)
                permutation_state = self.to_permutation_state(permutation_state)[:, None, :]
            else:  # permutation state == False
                permutation_state = 0.
            return self.sample(x=None, condition=condition+permutation_state)
        else:  # train
            if permutation_state is not None:
                permutation_state = self.to_permutation_state(permutation_state)[:, None, :]
            else:  # not use permutation state
                permutation_state = 0.
            # Given condition c and ground truth token x, compute loss
            c = self.model(output_shape, condition+permutation_state)
            assert c.shape[-1] == x_0.shape[-1], "d_model should be equal to dim_per_token"
            # preprocess nan to zero
            mask = torch.isnan(x_0)
            x_0 = torch.nan_to_num(x_0, 0.)
            # get the gradient
            loss = F.mse_loss(c, x_0, reduction="none")
            loss[mask] = torch.nan
            return loss.nanmean()

    @torch.no_grad()
    def sample(self, x=None, condition=None):
        z = self.model([1, self.sequence_length, self.config["d_model"]], condition)
        return z

    @property
    def device(self):
        return next(self.parameters()).device




class MambaDiffusion(ModelDiffusion):
    def __init__(self, sequence_length, positional_embedding):
        super().__init__(sequence_length=sequence_length)
        MambaModel.config = self.config
        self.model = MambaModel(positional_embedding=positional_embedding)


class TransformerDiffusion(ModelDiffusion):
    def __init__(self, sequence_length, positional_embedding):
        super().__init__(sequence_length=sequence_length)
        TransformerModel.config = self.config
        self.model = TransformerModel(positional_embedding=positional_embedding)


class LstmDiffusion(ModelDiffusion):
    def __init__(self, sequence_length, positional_embedding):
        super().__init__(sequence_length=sequence_length)
        LstmModel.config = self.config
        self.model = LstmModel(positional_embedding=positional_embedding)


class GMLPDiffusion(ModelDiffusion):
    def __init__(self, sequence_length, positional_embedding):
        super().__init__(sequence_length=sequence_length)
        GMLPModel.config = self.config
        self.model = GMLPModel(positional_embedding=positional_embedding)




class MambaMSELoss(ModelMSELoss):
    def __init__(self, sequence_length, positional_embedding):
        super().__init__(sequence_length=sequence_length)
        MambaModel.config = self.config
        self.model = MambaModel(positional_embedding=positional_embedding)




class ClassConditionMambaDiffusion(MambaDiffusion):
    def __init__(self, sequence_length, positional_embedding, input_class=10):
        super().__init__(sequence_length, positional_embedding)
        self.get_condition = nn.Sequential(
            nn.Linear(input_class, self.config["d_condition"]),
            nn.SiLU(),
        )  # to condition
        self.to_permutation_state = nn.Embedding(self.config["num_permutation"], self.config["d_model"])
        # condition module
        self.to_condition_linear = nn.Linear(self.config["d_condition"], self.config["d_model"])
        to_condition_gate = torch.zeros(size=(1, sequence_length, 1))
        to_condition_gate[:, -8:, :] = 1.
        self.register_buffer("to_condition_gate", to_condition_gate)
        # reset to_condition
        del self.to_condition
        self.to_condition = self._to_condition

    def forward(self, output_shape=None, x_0=None, condition=None, **kwargs):
        condition = self.get_condition(condition.to(self.device))
        return super().forward(output_shape=output_shape, x_0=x_0, condition=condition, **kwargs)

    def _to_condition(self, x):
        assert len(x.shape) == 3
        x = self.to_condition_linear(x)
        x = x * self.to_condition_gate
        return x


class ClassConditionMambaDiffusionFull(MambaDiffusion):
    def __init__(self, sequence_length, positional_embedding, input_class=10, init_noise_intensity=1e-4):
        super().__init__(sequence_length, positional_embedding)
        self.get_condition = nn.Sequential(
            nn.Linear(input_class, self.config["d_condition"]),
            nn.LayerNorm(self.config["d_condition"]),
        )  # to condition
        self.to_permutation_state = nn.Embedding(self.config["num_permutation"], self.config["d_model"])
        # condition module
        self.to_condition_linear = nn.Linear(self.config["d_condition"], self.config["d_model"])
        self.to_condition_conv = nn.Sequential(
            nn.Conv1d(1, sequence_length, 9, 1, 4),
            nn.GroupNorm(num_groups=1, num_channels=sequence_length),
            nn.Conv1d(sequence_length, sequence_length, 9, 1, 4),
        )  # [batch_size, sequence_length, d_model]
        # reset to_condition
        del self.to_condition

    def forward(self, output_shape=None, x_0=None, condition=None, **kwargs):
        if kwargs.get("pre_training"):
            self.to_condition = self._zero_condition
            condition = None
        else:  # train with condition
            self.to_condition = self._to_condition
            condition = self.get_condition(condition.to(self.device))
        return super().forward(output_shape=output_shape, x_0=x_0, condition=condition, **kwargs)

    def _to_condition(self, x):
        assert len(x.shape) == 3
        x = self.to_condition_linear(x)  # [batch_size, 1, d_model]
        x = self.to_condition_conv(x)  # [batch_size, sequence_length, d_model]
        return x

    def _zero_condition(self, x):
        return torch.zeros(size=(x.shape[0], self.sequence_length, self.config["d_model"]), device=x.device)
# ------------------------------------------------------------
# ------------------------------------------------------------          

class DatasetConditionMambaDiffusion(MambaDiffusion):
    def __init__(self, sequence_length, positional_embedding, cond_stage_config, cond_stage_trainable=False):
        super().__init__(sequence_length, positional_embedding)
        
        print("==> Initializing Dataset-Conditioned Mamba Diffusion...")
        
        # 动态地将主模型的 d_model 传递给条件模块的配置
        cond_stage_config['params']['d_model'] = self.config["d_model"]
        
        # 实例化我们的数据集编码器 (EmbedData)
        self.cond_stage_model = instantiate_from_config(cond_stage_config)

        # 根据需要冻结条件编码器
        if not cond_stage_trainable:
            print("==> Freezing conditioning stage model.")
            self.cond_stage_model.eval()
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False
                
        # 移除父类中简单的 to_condition 线性层，因为它将被我们的 cond_stage_model 完全取代
        if hasattr(self, 'to_condition'):
            del self.to_condition

    def forward(self, output_shape=None, x_0=None, condition=None, **kwargs):
        # 注意：这里的 'condition' 输入现在是数据集特征 (dataset_features)
        
        # 1. 使用 EmbedData 将数据集特征编码为 Mamba 兼容的条件张量 c
        #    形状: (batch_size, sequence_length, d_model)
        # print(f"[DEBUG DatasetConditionMambaDiffusion] Input condition device: {condition.device}")
        c = self.cond_stage_model(condition, self.sequence_length)
        # print(f"[DEBUG DatasetConditionMambaDiffusion] Output 'c' from EmbedData device: {c.device}")

        # 2. 调用父类的 forward 方法，但只传递我们精心准备的 c
        #    permutation_state 被忽略
        #    父类的 to_condition 不再被调用
        
        # --- 直接调用父类逻辑的核心部分 ---
        if kwargs.get("sample"):
            return self.sample(x=None, condition=c) # 直接将c用于采样
        else:
            # 训练时，直接将 c 传递给主干模型
            generated_c = self.model(output_shape, c)
            loss = self.criteria(x=x_0, c=generated_c, **kwargs)
            return loss