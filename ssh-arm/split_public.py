"""Split public/<class>/ into dataset/{train,val}/<class>/ at 80/20, seed 0."""
import random, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "public"
DST = ROOT / "dataset"
CLASSES = ("plastic", "paper", "metal", "glass")   # frozen
VAL_FRAC = 0.20
SEED = 0

if DST.exists():
    shutil.rmtree(DST)

rng = random.Random(SEED)
rows = []
for cls in CLASSES:
    files = sorted((SRC / cls).glob("*.jpg"))       # sorted => deterministic before shuffle
    rng.shuffle(files)
    n_val = round(len(files) * VAL_FRAC)
    parts = {"val": files[:n_val], "train": files[n_val:]}
    for split, group in parts.items():
        d = DST / split / cls
        d.mkdir(parents=True)
        for f in group:
            shutil.copy2(f, d / f.name)
    rows.append((cls, len(parts["train"]), len(parts["val"]), len(files)))

w = f"{'class':<9}{'train':>7}{'val':>7}{'total':>8}"
print(w); print("-" * len(w))
for cls, tr, va, tot in rows:
    print(f"{cls:<9}{tr:>7}{va:>7}{tot:>8}")
print("-" * len(w))
print(f"{'TOTAL':<9}{sum(r[1] for r in rows):>7}{sum(r[2] for r in rows):>7}{sum(r[3] for r in rows):>8}")
