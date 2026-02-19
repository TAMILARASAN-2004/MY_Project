# Multimodal Fake News Classifier

This repository contains a **multimodal fake news classifier** that can use:

- **Text** from a news article/headline
- **Image** attached to the article
- **Video** clip frames

The model builds modality-specific features and then trains a single classifier on the fused representation.

## Features

- Text embeddings with TF-IDF (`scikit-learn`)
- Image embeddings using a pretrained ResNet-18 (`torchvision`)
- Video embeddings by sampling frames and averaging frame embeddings
- Fusion + classification with logistic regression
- Simple command-line training and prediction workflow

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data format

Training data should be a CSV with these columns:

- `text`: news text
- `image_path`: path to image file (optional)
- `video_path`: path to video file (optional)
- `label`: `0` for real, `1` for fake

Example:

```csv
text,image_path,video_path,label
"Breaking news headline",data/img1.jpg,data/vid1.mp4,1
"Official statement from ministry",data/img2.jpg,,0
```

## Train

```bash
python fake_news_classifier.py train --data train.csv --model-out model.joblib
```

## Predict

```bash
python fake_news_classifier.py predict \
  --model-in model.joblib \
  --text "Shocking claim goes viral" \
  --image data/sample.jpg \
  --video data/sample.mp4
```

The output is:

- predicted class (`0=real, 1=fake`)
- probability of fake news

## Notes

- Missing modalities are supported; zeros are used if image/video are unavailable.
- Video support requires OpenCV (`opencv-python`).
- Pretrained weights for image encoding may download on first run.
