def merge_events(shards):
    """Candidate control that gets stable ordering right by buffering all inputs."""
    events = [event for shard in shards for event in shard]
    yield from sorted(events, key=lambda event: event[0])
