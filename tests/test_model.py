import torch

from fake_news_classifier.model import MultiModalFakeNewsClassifier


def test_multimodal_forward_shape():
    model = MultiModalFakeNewsClassifier(vocab_size=500, hidden_dim=64, num_classes=2)
    token_ids = torch.randint(1, 500, (2, 32))
    images = torch.rand(2, 3, 64, 64)
    videos = torch.rand(2, 4, 3, 64, 64)

    logits = model(token_ids, images, videos)

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
