"""Single source of truth for the waste-class -> bin -> cube-colour identity chain.

Every other module imports these names. Nothing else defines a class name, a bin
number, or a colour string. The two hand-written dicts below are the only place a
mapping is authored; the reverse directions are derived by inversion so the two
directions can never disagree.

The asserts run at import time on purpose. A broken mapping should kill the
process at startup, not quietly route glass into the plastic bin.
"""

# --- authored mappings (the only hand-written identity in the project) ---------

CLASS_TO_BIN = {"plastic": 1, "paper": 2, "metal": 3, "glass": 4}
BIN_TO_COLOR = {1: "red", 2: "green", 3: "blue", 4: "yellow"}

# --- derived mappings (never hand-write these) ---------------------------------

BIN_TO_CLASS = {bin_id: cls for cls, bin_id in CLASS_TO_BIN.items()}
COLOR_TO_BIN = {color: bin_id for bin_id, color in BIN_TO_COLOR.items()}

# Mirrors ssh-waste-segregation-/config.py:75. Duplicated as a literal so this
# module stays import-free and usable before the teammate path shim exists.
# Cross-check the real tuple at startup with verify_supported_colors().
TEAMMATE_SUPPORTED_COLORS = ("red", "green", "blue", "yellow")


# --- import-time integrity checks ---------------------------------------------

# Bijective: inversion must not have collapsed two keys onto one value.
assert len(BIN_TO_CLASS) == len(CLASS_TO_BIN), (
    f"CLASS_TO_BIN is not bijective — two classes share a bin: {CLASS_TO_BIN}"
)
assert len(COLOR_TO_BIN) == len(BIN_TO_COLOR), (
    f"BIN_TO_COLOR is not bijective — two bins share a colour: {BIN_TO_COLOR}"
)

# Total: every class must reach a bin, and every bin must reach a colour, or the
# class -> bin -> colour lookup in detect.py raises KeyError mid-run.
assert set(CLASS_TO_BIN.values()) == set(BIN_TO_COLOR), (
    "bin sets disagree — CLASS_TO_BIN targets "
    f"{sorted(set(CLASS_TO_BIN.values()))} but BIN_TO_COLOR defines "
    f"{sorted(BIN_TO_COLOR)}"
)

# The colours we ask for must be colours the teammate's detector can segment.
_unknown = set(BIN_TO_COLOR.values()) - set(TEAMMATE_SUPPORTED_COLORS)
assert not _unknown, (
    f"BIN_TO_COLOR uses colour(s) {sorted(_unknown)} that are not in the "
    f"teammate's SUPPORTED_COLORS {TEAMMATE_SUPPORTED_COLORS}"
)
del _unknown


# --- startup verifiers --------------------------------------------------------


def verify_model_names(names: dict) -> None:
    """Assert the loaded model's classes are exactly the classes we bin.

    Call this once at startup, before the capture loop, with ``model.names``.

    Ultralytics assigns class indices by sorting the dataset folder names, so a
    renamed, added, or dropped folder silently shifts every index. Without this
    check the pipeline keeps running and just bins everything wrong. ``nothing``
    is tolerated as an optional background class that has no bin.
    """
    if not isinstance(names, dict):
        raise TypeError(f"model.names must be a dict, got {type(names).__name__}")

    model_classes = set(names.values()) - {"nothing"}
    expected = set(CLASS_TO_BIN)
    if model_classes == expected:
        return

    missing = sorted(expected - model_classes)
    unexpected = sorted(model_classes - expected)
    detail = []
    if missing:
        detail.append(f"missing from model: {missing}")
    if unexpected:
        detail.append(f"unknown to contract.py: {unexpected}")
    raise ValueError(
        "model class names do not match CLASS_TO_BIN — refusing to run, since "
        "a class-index shift would mis-bin every item. "
        + "; ".join(detail)
        + f". model.names={names}, contract expects {sorted(expected)}"
    )


def verify_supported_colors(supported) -> None:
    """Cross-check the teammate's real SUPPORTED_COLORS against our mirror.

    Guards against their config drifting away from the literal above. Call it
    from the import shim once their config is actually importable.
    """
    if set(supported) != set(TEAMMATE_SUPPORTED_COLORS):
        raise ValueError(
            "teammate SUPPORTED_COLORS changed — contract.py mirrors "
            f"{TEAMMATE_SUPPORTED_COLORS} but their config now says "
            f"{tuple(supported)}. Update TEAMMATE_SUPPORTED_COLORS and re-check "
            "BIN_TO_COLOR."
        )
    unknown = set(BIN_TO_COLOR.values()) - set(supported)
    if unknown:
        raise ValueError(
            f"BIN_TO_COLOR uses colour(s) {sorted(unknown)} the teammate's "
            f"detector cannot segment (supports {tuple(supported)})"
        )


if __name__ == "__main__":
    print("CLASS_TO_BIN  ", CLASS_TO_BIN)
    print("BIN_TO_CLASS  ", BIN_TO_CLASS, "(derived)")
    print("BIN_TO_COLOR  ", BIN_TO_COLOR)
    print("COLOR_TO_BIN  ", COLOR_TO_BIN, "(derived)")
    print()
    for cls in sorted(CLASS_TO_BIN, key=CLASS_TO_BIN.get):
        b = CLASS_TO_BIN[cls]
        print(f"  {cls:<8} -> bin {b} -> expect {BIN_TO_COLOR[b]} cube")
