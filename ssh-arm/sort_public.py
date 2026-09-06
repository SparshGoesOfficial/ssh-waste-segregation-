"""Map the Kaggle 12-class garbage set into public/<class>/ for the 4-class baseline."""
import shutil, sys
from pathlib import Path

SRC = Path("/Users/adityasachan/.cache/kagglehub/datasets/mostafaabla/"
           "garbage-classification/versions/1/garbage_classification")
DST = Path(__file__).resolve().parent / "public"

# Frozen class names. Source folder -> target class, or None to drop.
MAP = {
    "plastic":     "plastic",
    "paper":       "paper",
    "cardboard":   "paper",
    "metal":       "metal",
    "green-glass": "glass",
    "brown-glass": "glass",
    "white-glass": "glass",
    "biological":  None,
    "clothes":     None,
    "shoes":       None,
    "battery":     None,   # dataset folder is 'battery', not 'batteries'
    "trash":       None,
}

found = {p.name for p in SRC.iterdir() if p.is_dir()}
if found != set(MAP):
    sys.exit(f"source folders changed\n  unmapped: {found - set(MAP)}\n  missing: {set(MAP) - found}")

if DST.exists():
    shutil.rmtree(DST)
for c in ("plastic", "paper", "metal", "glass"):
    (DST / c).mkdir(parents=True)

copied, dropped = {}, {}
for folder, cls in MAP.items():
    files = sorted((SRC / folder).glob("*.jpg"))
    if cls is None:
        dropped[folder] = len(files)
        continue
    for f in files:
        out = DST / cls / f.name
        assert not out.exists(), f"filename collision: {out}"
        shutil.copy2(f, out)
    copied[folder] = (cls, len(files))

print("copied:")
for folder, (cls, n) in copied.items():
    print(f"  {folder:<12} -> {cls:<8} {n:>5}")
print("dropped:")
for folder, n in dropped.items():
    print(f"  {folder:<12} {n:>5}")

print("\nper-class counts in public/:")
tot = 0
for c in ("plastic", "paper", "metal", "glass"):
    n = len(list((DST / c).glob("*.jpg")))
    print(f"  {c:<8} {n:>5}")
    tot += n
print(f"  {'TOTAL':<8} {tot:>5}")
