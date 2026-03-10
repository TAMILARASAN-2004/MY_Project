import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from fake_news_classifier.dataset import MultiModalNewsDataset, NewsExample
from fake_news_classifier.model import MultiModalFakeNewsClassifier


def build_synthetic_dataset(size: int = 100, vocab_size: int = 10000):
    examples = []
    for i in range(size):
        token_ids = torch.randint(1, vocab_size, (64,)).tolist()
        image = torch.rand(3, 128, 128)
        video = torch.rand(8, 3, 128, 128)
        label = i % 2
        examples.append(NewsExample(token_ids=token_ids, image=image, video=video, label=label))
    return MultiModalNewsDataset(examples)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_synthetic_dataset(size=args.samples, vocab_size=args.vocab_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MultiModalFakeNewsClassifier(vocab_size=args.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            images = batch["image"].to(device)
            videos = batch["video"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(token_ids, images, videos)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {running_loss / len(loader):.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multimodal fake news classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="multimodal_fake_news.pt")
    train(parser.parse_args())
