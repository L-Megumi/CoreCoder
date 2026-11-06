def decode_chunks(chunks):
    """Candidate control that fixes split scalars but consumes all input eagerly."""
    text = b"".join(chunks).decode("utf-8")
    if text:
        yield text
