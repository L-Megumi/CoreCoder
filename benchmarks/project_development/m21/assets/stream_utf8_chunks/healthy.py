import codecs


def decode_chunks(chunks):
    """Yield nonempty pieces of strict UTF-8 text from a single-pass byte stream."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    for chunk in chunks:
        text = decoder.decode(chunk)
        if text:
            yield text
    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail
