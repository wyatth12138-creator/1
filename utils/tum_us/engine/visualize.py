"""Small, deterministic prediction panels for experiment review."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def save_prediction_panels(output: dict, batch, directory: Path, prefix: str, maximum: int = 4) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    from tum_us.model.losses import supervised_mask_logits
    logits = supervised_mask_logits(output, batch).detach().float().cpu()
    media = batch.media.detach().float().cpu()
    masks = batch.masks.detach().float().cpu()
    valid = batch.mask_valid.detach().cpu().bool()
    probs = output["class_logits"].detach().float().softmax(-1).cpu()
    written = 0
    for b in range(len(batch.metadata)):
        annotated = torch.nonzero(valid[b], as_tuple=False)
        if len(annotated):
            t, q = annotated[0].tolist()
        else:
            t, q = int(torch.nonzero(batch.frame_valid[b].detach().cpu(), as_tuple=False)[0]), 0
        view = int(batch.seg_view_ids[b]) if hasattr(batch, "seg_view_ids") else 0
        # Contract is [B,T,V,C,H,W], including V=1 for single-view samples.
        frame = media[b, t, view] if media.ndim == 6 else media[b, t]
        array = frame.permute(1, 2, 0).numpy()
        if array.min() < 0:  # normalized [-1,1] display fallback
            array = (array + 1.0) * 0.5
        array = np.uint8(np.clip(array, 0, 1) * 255)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        size = masks.shape[-2:]
        prediction = logits[b, t, q]
        if prediction.shape != size:
            prediction = F.interpolate(prediction[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]
        pred = prediction.sigmoid().numpy() >= 0.5
        truth = masks[b, t, q].numpy() >= 0.5
        original = np.asarray(Image.fromarray(array).resize((size[1], size[0])))
        panels = []
        for title, mask, color in (("Input", None, None), ("Ground truth", truth if len(annotated) else None, [0, 210, 110]),
                                   ("Prediction", pred, [245, 90, 70])):
            rgb = original.copy()
            if mask is not None:
                rgb[mask] = np.uint8(0.6 * rgb[mask] + 0.4 * np.asarray(color))
            image = Image.new("RGB", (size[1], size[0] + 30), "white")
            image.paste(Image.fromarray(rgb), (0, 30))
            ImageDraw.Draw(image).text((8, 8), title, fill="black")
            panels.append(image)
        canvas = Image.new("RGB", (size[1] * 3, size[0] + 65), "white")
        for index, panel in enumerate(panels):
            canvas.paste(panel, (index * size[1], 0))
        text = f"P(class 1)={float(probs[b, 1]):.4f}; retained={output.get('memory_indices', [])[b]}"
        ImageDraw.Draw(canvas).text((8, size[0] + 40), text, fill="black")
        canvas.save(directory / f"{prefix}_{written:03d}.png")
        written += 1
        if written >= maximum:
            break
    return written
