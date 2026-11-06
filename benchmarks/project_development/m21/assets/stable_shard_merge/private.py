def f2p_stable_interleaved(module):
    shards = [[(0, "x"), (2, "z"), (2, "b")], [], [(1, "p"), (2, "a")], [(2, "m")]]
    assert list(module.merge_events(shards)) == [
        (0, "x"), (1, "p"), (2, "z"), (2, "b"), (2, "a"), (2, "m")]


def f2p_opaque_payloads(module):
    payloads = [object() for _ in range(6)]
    shards = [[(9, p) for p in payloads[:2]], [(9, p) for p in payloads[2:4]],
              [(9, p) for p in payloads[4:]]]
    assert [p for _, p in module.merge_events(shards)] == payloads


def p2p_sorted_distinct(module):
    shards = [[(-2, 1), (1, None)], [(0, "a"), (8, {})]]
    assert list(module.merge_events(shards)) == [shards[0][0], shards[1][0], shards[0][1], shards[1][1]]


def p2p_empty_and_single(module):
    assert list(module.merge_events([[], []])) == []
    events = [(1, 7), (2, 9)]
    assert list(module.merge_events([iter(events)])) == events


def p2p_one_head_per_shard(module):
    def stream(first):
        yield first, "payload"
        raise AssertionError("Read second event before returning first")

    assert next(iter(module.merge_events([stream(1), stream(2)]))) == (1, "payload")
