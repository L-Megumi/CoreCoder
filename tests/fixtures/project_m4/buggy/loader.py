"""Public contract: features is a sequence; preserve values and ordering."""


def load_values(record):
    return tuple(record["payload"])


def legacy_identity(value):
    return value
