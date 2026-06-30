"""
SceneForge: Dual Encoder + Cross-Modal Attention Fusion

RGB encoder:   Swin-T (pretrained ImageNet)
Depth encoder: ResNet34 (pretrained, adapted for single-channel input)
Fusion:        Bidirectional cross-modal attention
Detection:     DETR-style transformer decoder with amodal + occlusion heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models import resnet34, ResNet34_Weights


# ─────────────────────────────────────────────
# 1. RGB ENCODER (Swin-T)
# ─────────────────────────────────────────────
class RGBEncoder(nn.Module):
    """Extracts multi-scale RGB features. Output: (B, 768, H/32, W/32)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        swin = swin_t(weights=weights)

        self.features = swin.features
        self.norm = swin.norm
        self.out_channels = 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)            # (B, H/32, W/32, 768)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)        # (B, 768, H/32, W/32)
        return x


# ─────────────────────────────────────────────
# 2. DEPTH ENCODER (ResNet34)
# ─────────────────────────────────────────────
class DepthEncoder(nn.Module):
    """Encodes single-channel depth maps. Output: (B, 512, H/32, W/32)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet34(weights=weights)

        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.out_channels = 512

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        depth_3ch = depth.repeat(1, 3, 1, 1)   # replicate 1ch -> 3ch for pretrained weights
        return self.encoder(depth_3ch)


# ─────────────────────────────────────────────
# 3. CROSS-MODAL ATTENTION FUSION
# ─────────────────────────────────────────────
class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-attention fusion.
    RGB queries depth (depth gates attention in occluded regions).
    Depth queries RGB (RGB provides texture/colour context).
    """

    def __init__(
        self,
        rgb_channels: int = 768,
        depth_channels: int = 512,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.rgb_proj   = nn.Conv2d(rgb_channels, d_model, kernel_size=1)
        self.depth_proj = nn.Conv2d(depth_channels, d_model, kernel_size=1)

        self.rgb_to_depth_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.depth_to_rgb_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.norm_rgb   = nn.LayerNorm(d_model)
        self.norm_depth = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, rgb_feat: torch.Tensor, depth_feat: torch.Tensor) -> torch.Tensor:
        B, _, H, W = rgb_feat.shape

        if depth_feat.shape[-2:] != rgb_feat.shape[-2:]:
            depth_feat = F.interpolate(depth_feat, size=rgb_feat.shape[-2:], mode='bilinear', align_corners=False)

        rgb   = self.rgb_proj(rgb_feat)
        depth = self.depth_proj(depth_feat)

        rgb_seq   = rgb.flatten(2).permute(0, 2, 1)
        depth_seq = depth.flatten(2).permute(0, 2, 1)

        rgb_attended, _   = self.rgb_to_depth_attn(query=rgb_seq, key=depth_seq, value=depth_seq)
        depth_attended, _ = self.depth_to_rgb_attn(query=depth_seq, key=rgb_seq, value=rgb_seq)

        rgb_out   = self.norm_rgb(rgb_seq + self.dropout(rgb_attended))
        depth_out = self.norm_depth(depth_seq + self.dropout(depth_attended))

        fused_seq = torch.cat([rgb_out, depth_out], dim=-1)
        fused_seq = self.fusion_proj(fused_seq)

        fused = fused_seq.permute(0, 2, 1).reshape(B, self.d_model, H, W)
        return fused


# ─────────────────────────────────────────────
# 4. POSITIONAL ENCODING
# ─────────────────────────────────────────────
class PositionalEncoding2D(nn.Module):
    """Learnable 2D positional encoding."""

    def __init__(self, d_model: int, max_h: int = 50, max_w: int = 50):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, d_model // 2)
        self.col_embed = nn.Embedding(max_w, d_model // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        rows = torch.arange(H, device=x.device)
        cols = torch.arange(W, device=x.device)
        row_emb = self.row_embed(rows).unsqueeze(1).expand(H, W, -1)
        col_emb = self.col_embed(cols).unsqueeze(0).expand(H, W, -1)
        pos = torch.cat([row_emb, col_emb], dim=-1)
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        return x + pos


# ─────────────────────────────────────────────
# 5. DETR DETECTION HEAD
# ─────────────────────────────────────────────
class DETRDetectionHead(nn.Module):
    """
    Transformer decoder over fused features.
    Predicts: class, visible bbox, amodal bbox, occlusion score.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_decoder_layers: int = 6,
        num_queries: int = 100,
        num_classes: int = 40,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model

        self.query_embed = nn.Embedding(num_queries, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1 for no-object
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4), nn.Sigmoid(),
        )
        self.amodal_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4), nn.Sigmoid(),
        )
        self.occlusion_head = nn.Sequential(
            nn.Linear(d_model, 1), nn.Sigmoid(),
        )

    def forward(self, fused: torch.Tensor) -> dict:
        B, C, H, W = fused.shape
        memory = fused.flatten(2).permute(0, 2, 1)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        decoded = self.transformer_decoder(tgt=queries, memory=memory)

        return {
            "logits":           self.class_head(decoded),
            "boxes":            self.bbox_head(decoded),
            "amodal_boxes":     self.amodal_head(decoded),
            "occlusion_scores": self.occlusion_head(decoded),
        }


# ─────────────────────────────────────────────
# 6. FULL SceneForge MODEL
# ─────────────────────────────────────────────
class SceneForge(nn.Module):
    """
    Full RGB-D occlusion-aware detection model.
    rgb, depth -> encoders -> cross-modal fusion -> DETR head -> predictions
    """

    def __init__(
        self,
        num_classes: int = 40,
        num_queries: int = 100,
        d_model: int = 256,
        num_heads: int = 8,
        num_decoder_layers: int = 6,
        pretrained_encoders: bool = True,
    ):
        super().__init__()
        self.rgb_encoder   = RGBEncoder(pretrained=pretrained_encoders)
        self.depth_encoder = DepthEncoder(pretrained=pretrained_encoders)
        self.fusion = CrossModalAttention(
            rgb_channels=self.rgb_encoder.out_channels,
            depth_channels=self.depth_encoder.out_channels,
            d_model=d_model, num_heads=num_heads,
        )
        self.pos_encoding = PositionalEncoding2D(d_model=d_model)
        self.detection_head = DETRDetectionHead(
            d_model=d_model, num_heads=num_heads,
            num_decoder_layers=num_decoder_layers,
            num_queries=num_queries, num_classes=num_classes,
        )

        self.depth_quality_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(self.depth_encoder.out_channels, 1), nn.Sigmoid(),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, depth_quality_threshold: float = 0.3) -> dict:
        rgb_feat   = self.rgb_encoder(rgb)
        depth_feat = self.depth_encoder(depth)

        depth_quality = self.depth_quality_head(depth_feat)

        # Graceful degradation: zero out depth features when quality is low
        quality_gate = (depth_quality > depth_quality_threshold).float().view(-1, 1, 1, 1)
        depth_feat_gated = depth_feat * quality_gate

        fused = self.fusion(rgb_feat, depth_feat_gated)
        fused = self.pos_encoding(fused)

        predictions = self.detection_head(fused)
        predictions["depth_quality"] = depth_quality
        return predictions


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = SceneForge(num_classes=40, num_queries=100, d_model=256, pretrained_encoders=True).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")

    rgb   = torch.randn(2, 3, 224, 224).to(device)
    depth = torch.randn(2, 1, 224, 224).to(device)

    with torch.no_grad():
        out = model(rgb, depth)

    print(f"logits:           {out['logits'].shape}")
    print(f"boxes:            {out['boxes'].shape}")
    print(f"amodal_boxes:     {out['amodal_boxes'].shape}")
    print(f"occlusion_scores: {out['occlusion_scores'].shape}")
    print(f"depth_quality:    {out['depth_quality'].shape}")
    print("Sanity check passed.")