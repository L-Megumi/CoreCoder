def public_equal_timestamp(module):
    shards = [[(4, {"seq": 1})], [(4, {"seq": 2})]]
    assert list(module.merge_events(shards)) == [*shards[0], *shards[1]]
