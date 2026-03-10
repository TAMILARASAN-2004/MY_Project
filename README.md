# Multimodal Fake News Classifier (Text + Image + Video)

This project provides a baseline deep learning pipeline for fake news classification using three modalities:
- **Text** (headline/article tokens)
- **Image** (associated news image)
- **Video** (short clip represented as a sequence of frames)

## Architecture

`MultiModalFakeNewsClassifier` has three encoders:
1. **TextEncoder**: Embedding + BiGRU
2. **ImageEncoder**: Lightweight CNN
3. **VideoEncoder**: Frame-level CNN + temporal BiGRU

The outputs are concatenated and passed through an MLP fusion head for binary classification (`real` vs `fake`).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train (demo on synthetic data)

```bash
python -m fake_news_classifier.train --epochs 3 --batch-size 4 --samples 100
```

### Inference

```bash
python -m fake_news_classifier.infer --model-path multimodal_fake_news.pt
```

### Run tests

```bash
pytest -q
```

## Integrating real data

Use `NewsExample` and `MultiModalNewsDataset` in `fake_news_classifier/dataset.py` and replace synthetic data generation in `train.py` with your own data loader:
- tokenized text ids
- resized image tensor `[3, H, W]`
- video tensor `[T, 3, H, W]`
- label (`0` for real, `1` for fake)
