from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import models, transforms


@dataclass
class MultimodalRecord:
    text: str
    image_path: Optional[str]
    video_path: Optional[str]


class MultimodalFakeNewsClassifier:
    def __init__(self, max_text_features: int = 5000, image_feature_dim: int = 512):
        self.text_vectorizer = TfidfVectorizer(max_features=max_text_features)
        self.image_feature_dim = image_feature_dim
        self.classifier = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.image_encoder = self._build_image_encoder().to(self.device)
        self.image_encoder.eval()

    def _build_image_encoder(self) -> torch.nn.Module:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)
        feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-1])
        return feature_extractor

    def _extract_single_image_feature(self, image_path: Optional[str]) -> np.ndarray:
        if not image_path:
            return np.zeros(self.image_feature_dim, dtype=np.float32)

        path = Path(image_path)
        if not path.exists():
            return np.zeros(self.image_feature_dim, dtype=np.float32)

        image = Image.open(path).convert("RGB")
        tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.image_encoder(tensor)

        return feat.squeeze().detach().cpu().numpy().astype(np.float32)

    def _extract_video_feature(self, video_path: Optional[str], max_frames: int = 12) -> np.ndarray:
        if not video_path:
            return np.zeros(self.image_feature_dim, dtype=np.float32)

        path = Path(video_path)
        if not path.exists():
            return np.zeros(self.image_feature_dim, dtype=np.float32)

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return np.zeros(self.image_feature_dim, dtype=np.float32)

        frame_features: list[np.ndarray] = []
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            capture.release()
            return np.zeros(self.image_feature_dim, dtype=np.float32)

        sample_indices = np.linspace(0, max(frame_count - 1, 0), num=min(max_frames, frame_count), dtype=int)
        selected = set(sample_indices.tolist())

        idx = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if idx in selected:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                tensor = self.image_transform(pil_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.image_encoder(tensor)
                frame_features.append(feat.squeeze().detach().cpu().numpy().astype(np.float32))
            idx += 1

        capture.release()

        if not frame_features:
            return np.zeros(self.image_feature_dim, dtype=np.float32)

        return np.mean(np.stack(frame_features, axis=0), axis=0)

    def build_features(self, records: Iterable[MultimodalRecord], fit_text: bool) -> np.ndarray:
        records_list = list(records)
        texts = [r.text or "" for r in records_list]

        if fit_text:
            text_features = self.text_vectorizer.fit_transform(texts)
        else:
            text_features = self.text_vectorizer.transform(texts)

        dense_text = text_features.toarray().astype(np.float32)
        image_features = np.stack([self._extract_single_image_feature(r.image_path) for r in records_list], axis=0)
        video_features = np.stack([self._extract_video_feature(r.video_path) for r in records_list], axis=0)

        return np.concatenate([dense_text, image_features, video_features], axis=1)

    def fit(self, records: Iterable[MultimodalRecord], labels: Iterable[int]) -> None:
        x = self.build_features(records, fit_text=True)
        y = np.array(list(labels), dtype=np.int64)
        self.classifier.fit(x, y)

    def predict_proba(self, records: Iterable[MultimodalRecord]) -> np.ndarray:
        x = self.build_features(records, fit_text=False)
        probs = self.classifier.predict_proba(x)
        return probs[:, 1]

    def predict(self, records: Iterable[MultimodalRecord]) -> np.ndarray:
        x = self.build_features(records, fit_text=False)
        return self.classifier.predict(x)

    def save(self, output_path: str) -> None:
        payload = {
            "text_vectorizer": self.text_vectorizer,
            "classifier": self.classifier,
            "image_feature_dim": self.image_feature_dim,
        }
        joblib.dump(payload, output_path)

    @classmethod
    def load(cls, model_path: str) -> "MultimodalFakeNewsClassifier":
        payload = joblib.load(model_path)
        model = cls(image_feature_dim=payload["image_feature_dim"])
        model.text_vectorizer = payload["text_vectorizer"]
        model.classifier = payload["classifier"]
        return model


def _read_records_from_dataframe(df: pd.DataFrame) -> list[MultimodalRecord]:
    return [
        MultimodalRecord(
            text=str(row.get("text", "")),
            image_path=str(row["image_path"]) if pd.notna(row.get("image_path", None)) else None,
            video_path=str(row["video_path"]) if pd.notna(row.get("video_path", None)) else None,
        )
        for _, row in df.iterrows()
    ]


def train(data_csv: str, model_out: str) -> None:
    df = pd.read_csv(data_csv)
    expected = {"text", "image_path", "video_path", "label"}
    missing = expected.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    records = _read_records_from_dataframe(df)
    labels = df["label"].astype(int).tolist()

    model = MultimodalFakeNewsClassifier()
    model.fit(records, labels)
    model.save(model_out)


def predict(model_in: str, text: str, image_path: Optional[str], video_path: Optional[str]) -> tuple[int, float]:
    model = MultimodalFakeNewsClassifier.load(model_in)
    record = MultimodalRecord(text=text, image_path=image_path, video_path=video_path)
    pred = int(model.predict([record])[0])
    prob_fake = float(model.predict_proba([record])[0])
    return pred, prob_fake


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal fake news classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--data", required=True, help="Path to CSV dataset")
    train_parser.add_argument("--model-out", required=True, help="Output model path")

    predict_parser = subparsers.add_parser("predict", help="Run prediction")
    predict_parser.add_argument("--model-in", required=True, help="Input saved model path")
    predict_parser.add_argument("--text", required=True, help="News text")
    predict_parser.add_argument("--image", default=None, help="Optional image path")
    predict_parser.add_argument("--video", default=None, help="Optional video path")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(data_csv=args.data, model_out=args.model_out)
        print(f"Model saved to {args.model_out}")
    else:
        pred, prob_fake = predict(
            model_in=args.model_in,
            text=args.text,
            image_path=args.image,
            video_path=args.video,
        )
        print(f"prediction={pred} fake_probability={prob_fake:.4f}")


if __name__ == "__main__":
    main()
