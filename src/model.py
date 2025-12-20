# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from transformers import PatchTSTModel


def freeze_module(m: nn.Module):
    """Freeze all parameters in a module (no gradient updates)."""
    for p in m.parameters():
        p.requires_grad = False


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    InfoNCE loss within a batch.

    Args:
        z1, z2: shape (B, D). Positive pairs are matched by the same index.
               Negatives are other samples within the batch.
        temperature: softmax temperature.

    Returns:
        Scalar contrastive loss.
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = (z1 @ z2.t()) / temperature  # (B, B)
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


class PatchTSTFrozenEncoder(nn.Module):
    """
    Use Hugging Face PatchTSTModel as a frozen encoder.

    - Input: (B, L), single-channel raw waveform
    - Internals:
        1) Resample to context_length (defaults to config.context_length, fallback=512)
        2) Repeat channels to num_input_channels (defaults to config.num_input_channels, fallback=7)
    - Output: (B, D) embedding (mean pooling over last_hidden_state)
    """
    def __init__(
        self,
        model_name: str = "ibm-granite/granite-timeseries-patchtst",
        cache_dir: str | None = None,
        target_len: int | None = None,
        target_channels: int | None = None,
    ):
        """
        Args:
            model_name: HF model name or local path.
            cache_dir: Optional HF cache directory (leave None for default; avoid hard-coded absolute paths in GitHub).
            target_len: Optional override for context length.
            target_channels: Optional override for number of input channels.
        """
        super().__init__()
        self.model = PatchTSTModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()
        freeze_module(self.model)

        cfg = self.model.config

        # Common PatchTST config fields: num_input_channels / context_length
        self.num_input_channels = int(
            getattr(cfg, "num_input_channels", 7) if target_channels is None else target_channels
        )
        self.context_length = int(
            getattr(cfg, "context_length", 512) if target_len is None else target_len
        )

        # Hidden size (different configs may use different attribute names)
        self.hidden_size = int(getattr(cfg, "hidden_size", getattr(cfg, "d_model", 768)))

    @torch.no_grad()
    def forward(self, x_1d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_1d: (B, L)

        Returns:
            emb: (B, D)
        """
        # 1) Resample to context_length (compress the whole sequence, no sliding window)
        # (B, L) -> (B, 1, L) -> (B, 1, T)
        x = x_1d.unsqueeze(1)
        x = F.interpolate(x, size=self.context_length, mode="linear", align_corners=False)

        # 2) Repeat channels to num_input_channels
        # (B, 1, T) -> (B, C, T) -> (B, T, C)
        x = x.repeat(1, self.num_input_channels, 1).transpose(1, 2).contiguous()

        out = self.model(past_values=x, return_dict=True)
        h = out.last_hidden_state  # typically (B, n_patches, hidden) but can vary by model/config

        # Pool across all non-batch, non-hidden dimensions to obtain (B, H)
        if h.dim() == 2:
            emb = h
        elif h.dim() == 3:
            # (B, T, H)
            emb = h.mean(dim=1)
        elif h.dim() == 4:
            # Common cases: (B, C, T, H) or (B, T, C, H)
            # Mean over all dims except batch and the last hidden dim
            emb = h.mean(dim=tuple(range(1, h.dim() - 1)))
        else:
            raise RuntimeError(f"Unexpected last_hidden_state dim={h.dim()}, shape={tuple(h.shape)}")

        return emb


class FusionTransformer(nn.Module):
    def __init__(
        self,
        emb_dim: int = 256,
        n_layers: int = 1,
        n_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        n_classes: int = 4,
    ):
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
        """
        Args:
            tokens: (B, 3, D) modality tokens

        Returns:
            logits: (B, n_classes)
        """
        B = tokens.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)  # (B, 4, D)
        x = self.tr(x)
        return self.head(x[:, 0])  # (B, n_classes)


class MultiModalEmotionModel(nn.Module):
    """
    Use a single shared PatchTSTFrozenEncoder for all three modalities (shared weights).
    The encoder is frozen. We only train:
      - to_emb (projection from encoder hidden size to emb_dim)
      - proj (contrastive projection head)
      - fuser (transformer fusion + classifier)
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

        # Map PatchTST hidden size to the desired embedding dimension (trainable)
        self.to_emb = nn.Linear(enc_dim, emb_dim)

        # Contrastive projection head (trainable)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )

        self.fuser = FusionTransformer(
            emb_dim=emb_dim,
            n_layers=1,
            n_heads=4,
            ff_dim=512,
            dropout=0.1,
            n_classes=n_classes,
        )

    def forward(self, batch: Dict[str, torch.Tensor], temperature: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: dict with keys {"ecg","eda","ppg"} and tensors shaped (B, L)
            temperature: InfoNCE temperature

        Returns:
            dict with:
              - logits: (B, n_classes)
              - lc: scalar contrastive loss
        """
        ecg, eda, ppg = batch["ecg"], batch["eda"], batch["ppg"]

        # Frozen PatchTST encoder
        with torch.no_grad():
            z_ecg = self.encoder(ecg)  # (B, enc_dim)
            z_eda = self.encoder(eda)
            z_ppg = self.encoder(ppg)

        # Trainable mapping to a unified emb_dim
        z_ecg = self.to_emb(z_ecg)
        z_eda = self.to_emb(z_eda)
        z_ppg = self.to_emb(z_ppg)

        # Contrastive learning on projected embeddings
        p_ecg = self.proj(z_ecg)
        p_eda = self.proj(z_eda)
        p_ppg = self.proj(z_ppg)

        lc = (
            info_nce(p_ecg, p_eda, temperature)
            + info_nce(p_ecg, p_ppg, temperature)
            + info_nce(p_eda, p_ppg, temperature)
        ) / 3.0

        tokens = torch.stack([z_ecg, z_eda, z_ppg], dim=1)  # (B, 3, emb_dim)
        logits = self.fuser(tokens)

        return {"logits": logits, "lc": lc}
