def decode_chunks(chunks):
    """Yield nonempty pieces of strict UTF-8 text from a single-pass byte stream."""
    for chunk in chunks:
        text = chunk.decode("utf-8")
        if text:
            yield text
