"""Public contract: total_features sums the features without changing its input."""
from loader import legacy_identity, load_values


def total_features(record):
    try:
        return sum(load_values(record))
    except KeyError as error:
        raise RuntimeError("record has no usable feature list") from error
