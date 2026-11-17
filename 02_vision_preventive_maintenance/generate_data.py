"""Generate synthetic camera images of machine parts for Vision-Based Preventive Maintenance (Use Case 2).

Creates 400 RGB images (64x64) split evenly into:
- normal/: Machined circular part with realistic lighting variation & camera noise
- defective/: Machined circular part with surface scratches, pitting, or edge-chip shape distortions
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def _create_base_part(rng: np.random.Generator, size: int = 64) -> np.ndarray:
    """Render a synthetic circular machine part with metallic shading and camera noise."""
    yy, xx = np.mgrid[0:size, 0:size]
    cx = size / 2.0 + rng.uniform(-2.5, 2.5)
    cy = size / 2.0 + rng.uniform(-2.5, 2.5)
    radius = size * rng.uniform(0.34, 0.39)
    inner_r = radius * rng.uniform(0.30, 0.38)

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Background conveyor belt (dark textured gray)
    bg_val = rng.uniform(35, 60)
    img = np.full((size, size, 3), bg_val, dtype=np.float32)

    # Part body (metallic blue-gray with radial gradient)
    part_mask = (dist <= radius) & (dist >= inner_r)
    grad = 1.0 - 0.25 * (dist / radius)
    base_color = np.array([75.0, 125.0, 195.0]) * rng.uniform(0.85, 1.15)
    for c in range(3):
        channel = img[:, :, c]
        channel[part_mask] = base_color[c] * grad[part_mask]

    # Subtle directional lighting & sensor noise
    light = rng.uniform(-12, 12) * ((xx - cx) / size)
    img += light[:, :, None]
    noise = rng.normal(0, rng.uniform(6.0, 12.0), size=(size, size, 3))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def _add_defect(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Inject one or more realistic surface/shape defects onto a machine part image."""
    size = img.shape[0]
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    defect_type = rng.choice(["scratch", "pitting", "edge_chip"])

    if defect_type == "scratch":
        x1, y1 = int(rng.integers(14, size - 14)), int(rng.integers(14, size - 14))
        angle = rng.uniform(0, np.pi)
        length = rng.uniform(12, 24)
        x2 = int(np.clip(x1 + length * np.cos(angle), 6, size - 6))
        y2 = int(np.clip(y1 + length * np.sin(angle), 6, size - 6))
        color = tuple(int(v) for v in rng.integers(190, 245, size=3))
        draw.line([(x1, y1), (x2, y2)], fill=color, width=int(rng.integers(1, 3)))

    elif defect_type == "pitting":
        n_pits = int(rng.integers(3, 7))
        for _ in range(n_pits):
            px = int(rng.integers(16, size - 16))
            py = int(rng.integers(16, size - 16))
            r = int(rng.integers(2, 4))
            dark = int(rng.integers(15, 50))
            draw.ellipse([px - r, py - r, px + r, py + r], fill=(dark, dark, dark + 5))

    else:  # edge_chip (shape distortion)
        angle = rng.uniform(0, 2 * np.pi)
        cx, cy = size / 2.0, size / 2.0
        r_edge = size * 0.36
        vx = cx + r_edge * np.cos(angle)
        vy = cy + r_edge * np.sin(angle)
        spread = 0.45
        p1 = (vx + 8 * np.cos(angle + spread), vy + 8 * np.sin(angle + spread))
        p2 = (vx + 8 * np.cos(angle - spread), vy + 8 * np.sin(angle - spread))
        p3 = (cx + (r_edge * 0.55) * np.cos(angle), cy + (r_edge * 0.55) * np.sin(angle))
        draw.polygon([p1, p2, p3], fill=(42, 42, 46))

    arr = np.array(pil_img, dtype=np.float32)
    arr += rng.normal(0, 5.0, size=arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8), str(defect_type)


def generate_vision_dataset(out_dir: Path, n_per_class: int = 200, size: int = 64, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic normal and defective part images and save metadata."""
    rng = np.random.default_rng(seed)
    normal_dir = out_dir / "normal"
    defective_dir = out_dir / "defective"
    normal_dir.mkdir(parents=True, exist_ok=True)
    defective_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(n_per_class):
        img_norm = _create_base_part(rng, size=size)
        norm_path = normal_dir / f"normal_{i:03d}.png"
        Image.fromarray(img_norm).save(norm_path)
        records.append({"filename": f"normal/normal_{i:03d}.png", "label": 0, "class_name": "normal", "defect_type": "none"})

        base_def = _create_base_part(rng, size=size)
        img_def, defect_type = _add_defect(base_def, rng)
        def_path = defective_dir / f"defective_{i:03d}.png"
        Image.fromarray(img_def).save(def_path)
        records.append({"filename": f"defective/defective_{i:03d}.png", "label": 1, "class_name": "defective", "defect_type": defect_type})

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "metadata.csv", index=False)
    return df


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "data" / "parts_dataset"
    df = generate_vision_dataset(dataset_dir, n_per_class=200, size=64, seed=42)
    print(f"Generated {len(df)} synthetic part images in: {dataset_dir}")
    print(df["defect_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
