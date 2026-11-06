def public_split_character(module):
    raw = "a€z".encode("utf-8")
    assert "".join(module.decode_chunks([raw[:2], raw[2:]])) == "a€z"
