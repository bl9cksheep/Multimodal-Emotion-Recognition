# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from transformers import PatchTSTModel


def freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    z1,z2: (B,D)，正样本是同index配对，负样本是batch内其他样本
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = (z1 @ z2.t()) / temperature  # (B,B)
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


class PatchTSTFrozenEncoder(nn.Module):
    """
    用 HF PatchTSTModel 作为 frozen encoder：
    - 输入: (B, L) 单通道原始波形
    - 内部: resample 到 context_length（默认取 config.context_length，否则 fallback=512）
            并复制到 num_input_channels（默认取 config.num_input_channels，否则 fallback=7）
    - 输出: (B, D) embedding（mean pooling last_hidden_state）
    """
    def __init__(
        self,
        model_name: str = "ibm-granite/granite-timeseries-patchtst",
        cache_dir: str | None = None,
        target_len: int | None = None,
        target_channels: int | None = None,
    ):
        super().__init__()
        self.model = PatchTSTModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()
        freeze_module(self.model)

        cfg = self.model.config

        # PatchTST 常见字段：num_input_channels / context_length
        self.num_input_channels = int(getattr(cfg, "num_input_channels", 7) if target_channels is None else target_channels)
        self.context_length = int(getattr(cfg, "context_length", 512) if target_len is None else target_len)

        # hidden size
        self.hidden_size = int(getattr(cfg, "hidden_size", getattr(cfg, "d_model", 768)))

    @torch.no_grad()
    def forward(self, x_1d: torch.Tensor) -> torch.Tensor:
        """
        x_1d: (B, L)
        return: (B, D)
        """
        # 1) resample 到 context_length（不滑窗，整段压缩）
        # (B,L) -> (B,1,L) -> (B,1,T)
        x = x_1d.unsqueeze(1)
        x = F.interpolate(x, size=self.context_length, mode="linear", align_corners=False)

        # 2) 复制通道到 num_input_channels
        # (B,1,T) -> (B,C,T) -> (B,T,C)
        x = x.repeat(1, self.num_input_channels, 1).transpose(1, 2).contiguous()

        out = self.model(past_values=x, return_dict=True)
        h = out.last_hidden_state               # (B, n_patches, hidden)
        # 统一把除了 batch 和 hidden 以外的维度都平均掉
        # 目标：emb -> (B, H)
        if h.dim() == 2:
            emb = h
        elif h.dim() == 3:
            # (B, T, H)
            emb = h.mean(dim=1)
        elif h.dim() == 4:
            # 常见情况： (B, C, T, H) 或 (B, T, C, H)
            # 我们把除 B 和最后一维 H 之外的维度全部 mean
            emb = h.mean(dim=tuple(range(1, h.dim()-1)))
        else:
            raise RuntimeError(f"Unexpected last_hidden_state dim={h.dim()}, shape={tuple(h.shape)}")

        return emb


class FusionTransformer(nn.Module):
    def __init__(self, emb_dim=256, n_layers=1, n_heads=4, ff_dim=512, dropout=0.1, n_classes=4):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, emb_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, n_classes))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B,3,D)
        B = tokens.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)    # (B,4,D)
        x = self.tr(x)
        return self.head(x[:, 0])             # (B,4)


class MultiModalEmotionModel(nn.Module):
    """
    三模态都用同一个 PatchTSTFrozenEncoder（共享权重，节省显存&保证同空间）。
    encoder frozen，只训练：
    - proj（对比学习）
    - fuser（transformer融合）
    """
    def __init__(
        self,
        patchtst_name: str = "ibm-granite/granite-timeseries-patchtst",
        cache_dir: str | None = None,
        emb_dim: int = 256,
        n_classes: int = 4,
    ):
        super().__init__()

        self.encoder = PatchTSTFrozenEncoder(model_name=patchtst_name, cache_dir=cache_dir)
        enc_dim = self.encoder.hidden_size

        # 把 PatchTST 的 hidden 映射到你想要的 embedding 维度（可训练）
        self.to_emb = nn.Linear(enc_dim, emb_dim)

        # 对比学习投影头（可训练）
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )

        self.fuser = FusionTransformer(emb_dim=emb_dim, n_layers=1, n_heads=4, ff_dim=512, dropout=0.1, n_classes=n_classes)

    def forward(self, batch: Dict[str, torch.Tensor], temperature: float = 0.1) -> Dict[str, torch.Tensor]:
        ecg, eda, ppg = batch["ecg"], batch["eda"], batch["ppg"]

        # frozen PatchTST encoder
        with torch.no_grad():
            z_ecg = self.encoder(ecg)  # (B, enc_dim)
            z_eda = self.encoder(eda)
            z_ppg = self.encoder(ppg)

        # 映射到统一 emb_dim（可训练）
        z_ecg = self.to_emb(z_ecg)
        z_eda = self.to_emb(z_eda)
        z_ppg = self.to_emb(z_ppg)

        # contrastive on projected embeddings
        p_ecg = self.proj(z_ecg)
        p_eda = self.proj(z_eda)
        p_ppg = self.proj(z_ppg)

        lc = (
            info_nce(p_ecg, p_eda, temperature)
            + info_nce(p_ecg, p_ppg, temperature)
            + info_nce(p_eda, p_ppg, temperature)
        ) / 3.0

        tokens = torch.stack([z_ecg, z_eda, z_ppg], dim=1)  # (B,3,emb_dim)
        logits = self.fuser(tokens)

        return {"logits": logits, "lc": lc}
