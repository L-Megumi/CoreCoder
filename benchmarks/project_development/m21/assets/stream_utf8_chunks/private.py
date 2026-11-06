def f2p_single_byte_partitions(module):
    expected = "雪🙂é-A"
    raw = expected.encode("utf-8")
    assert "".join(module.decode_chunks(bytes([value]) for value in raw)) == expected


def f2p_every_binary_split(module):
    expected = "é/水/𐍈"
    raw = expected.encode("utf-8")
    for split in range(len(raw) + 1):
        output = list(module.decode_chunks([raw[:split], b"", raw[split:]]))
        assert all(isinstance(part, str) and part for part in output)
        assert "".join(output) == expected


def p2p_strict_errors(module):
    for broken in ([b"\xe4\xb8"], [b"ascii", b"\xff"], [b"\xe0", b"\x80\x80"]):
        try:
            list(module.decode_chunks(broken))
        except UnicodeDecodeError:
            continue
        raise AssertionError("Malformed or incomplete UTF-8 was accepted")


def p2p_empty_chunks(module):
    assert list(module.decode_chunks([b"", b"", b""])) == []


def p2p_streaming_backpressure(module):
    def source():
        yield b"ready"
        raise AssertionError("Consumed next input before yielding complete text")

    assert next(iter(module.decode_chunks(source()))) == "ready"
