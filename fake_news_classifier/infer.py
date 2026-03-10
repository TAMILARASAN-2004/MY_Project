import argparse

import torch

from fake_news_classifier.model import MultiModalFakeNewsClassifier


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalFakeNewsClassifier(vocab_size=args.vocab_size)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    token_ids = torch.randint(1, args.vocab_size, (1, args.max_text_len)).to(device)
    image = torch.rand(1, 3, 128, 128).to(device)
    video = torch.rand(1, args.frames, 3, 128, 128).to(device)

    with torch.no_grad():
        logits = model(token_ids, image, video)
        probs = torch.softmax(logits, dim=-1)
        label = torch.argmax(probs, dim=-1).item()

    print({"prediction": "fake" if label == 1 else "real", "probabilities": probs.cpu().tolist()[0]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for multimodal fake news classifier")
    parser.add_argument("--model-path", type=str, default="multimodal_fake_news.pt")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--max-text-len", type=int, default=128)
    parser.add_argument("--frames", type=int, default=8)
    predict(parser.parse_args())
