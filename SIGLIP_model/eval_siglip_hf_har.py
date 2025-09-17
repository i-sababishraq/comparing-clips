#!/usr/bin/env python3
"""
Zero-shot evaluation on HAR dataset using Hugging Face SigLIP.

- Model: google/siglip-base-patch16-224
- Prompts: "a photo of {class_name}"
"""

import os
import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, SiglipModel


def list_classes(split_dir: Path) -> List[str]:
    classes = [d.name for d in split_dir.iterdir() if d.is_dir()]
    classes.sort()
    return classes


def iter_images(split_dir: Path):
    for cls in sorted([d for d in split_dir.iterdir() if d.is_dir()], key=lambda p: p.name):
        for f in cls.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                yield f, cls.name


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Ensure HF cache is writable in this environment
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(__file__).parent / "hf_cache"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = SiglipModel.from_pretrained(args.model_name).to(device)
    model.eval()

    split_dir = Path(args.data_dir) / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split not found: {split_dir}")

    class_names = list_classes(split_dir)
    if len(class_names) == 0:
        raise RuntimeError(f"No classes found under {split_dir}")

    # Prepare class prompts and text embeddings
    prompts = [f"a photo of {c}" for c in class_names]
    text_inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
    try:
        text_embeds = model.get_text_features(**text_inputs)
    except AttributeError:
        out_t = model(**text_inputs)
        text_embeds = out_t.text_embeds if hasattr(out_t, "text_embeds") else out_t[1]
    text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)

    # Iterate images in batches
    image_paths, image_labels = [], []
    for path, cls in iter_images(split_dir):
        image_paths.append(path)
        image_labels.append(class_names.index(cls))

    total = len(image_paths)
    correct = 0
    for i in tqdm(range(0, total, args.batch_size), desc="Zero-shot Eval"):
        batch_paths = image_paths[i:i+args.batch_size]
        batch_labels = image_labels[i:i+args.batch_size]

        pil_images = []
        for p in batch_paths:
            try:
                pil_images.append(Image.open(p).convert("RGB"))
            except Exception:
                # Skip unreadable images; keep alignment by inserting a black image
                pil_images.append(Image.new("RGB", (224, 224), color=(0, 0, 0)))

        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        try:
            image_embeds = model.get_image_features(**inputs)
        except AttributeError:
            out_i = model(**inputs)
            image_embeds = out_i.image_embeds if hasattr(out_i, "image_embeds") else out_i[0]
        image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)

        # Similarity and prediction
        logits = image_embeds @ text_embeds.t()
        preds = logits.argmax(dim=1).detach().cpu().tolist()
        correct += sum(int(p == y) for p, y in zip(preds, batch_labels))

    acc = correct / max(1, total)
    print(f"Classes: {len(class_names)}  Samples: {total}  Zero-shot Acc: {acc:.4f}")


if __name__ == "__main__":
    main()


