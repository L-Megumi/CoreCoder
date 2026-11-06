from heapq import merge


def merge_events(shards):
    """Merge sorted (timestamp, payload) streams in stable shard order."""
    yield from merge(*shards, key=lambda event: event[0])
