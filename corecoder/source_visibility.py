"""Conservative source ranges backed by complete lines in a tool's excerpt."""


def visible_end_line(span):
    """Do not treat requested-but-truncated lines as observed source.

    A character-truncated last line is excluded. Legacy artifacts without a
    char_truncated flag are handled conservatively when fewer lines are present.
    This describes the original tool excerpt, not later context projections.
    """
    start, end = span.get("start_line", 0), span.get("end_line", -1)
    content = span.get("content")
    if (not isinstance(content, str) or isinstance(start, bool) or not isinstance(start, int)
            or isinstance(end, bool) or not isinstance(end, int) or start < 1 or end < start):
        return 0
    expected = end - start + 1
    lines = content.count("\n") + 1
    cut = span.get("char_truncated") or lines < expected
    count = min(expected, lines - 1 if cut else lines)
    return start + count - 1
