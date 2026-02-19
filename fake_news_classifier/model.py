import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        _, hidden = self.gru(embedded)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.projection(hidden)


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.cnn(images).flatten(1)
        return self.projection(features)


class VideoEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.frame_encoder = ImageEncoder(in_channels=in_channels, hidden_dim=hidden_dim)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        # videos: [batch, frames, channels, height, width]
        b, t, c, h, w = videos.shape
        frames = videos.view(b * t, c, h, w)
        frame_embeddings = self.frame_encoder(frames).view(b, t, -1)
        _, hidden = self.temporal(frame_embeddings)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.projection(hidden)


class MultiModalFakeNewsClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.image_encoder = ImageEncoder(hidden_dim=hidden_dim)
        self.video_encoder = VideoEncoder(hidden_dim=hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        token_ids: torch.Tensor,
        images: torch.Tensor,
        videos: torch.Tensor,
    ) -> torch.Tensor:
        text_features = self.text_encoder(token_ids)
        image_features = self.image_encoder(images)
        video_features = self.video_encoder(videos)

        fused = torch.cat([text_features, image_features, video_features], dim=-1)
        fused = self.fusion(fused)
        return self.classifier(fused)
