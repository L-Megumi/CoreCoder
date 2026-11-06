from heapq import heappop, heappush


def merge_events(shards):
    """Merge sorted (timestamp, payload) streams in stable shard order."""
    heap = []
    for shard in shards:
        iterator = iter(shard)
        event = next(iterator, None)
        if event is not None:
            heappush(heap, (event[0], event[1], iterator))
    while heap:
        timestamp, payload, iterator = heappop(heap)
        yield timestamp, payload
        event = next(iterator, None)
        if event is not None:
            heappush(heap, (event[0], event[1], iterator))
