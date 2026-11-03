from corecoder.evidence import extract_evidence

def test_extract_evidence_normal():
    text = """
    This is a normal line.
    Another normal line.
    """
    evidence = extract_evidence(text)
    assert evidence == []


def test_extract_evidence_error():
    text = """
    RuntimeError
    """
    evidence = extract_evidence(text)
    assert len(evidence) == 1
    assert evidence[0]["type"] == "ERROR"
    assert evidence[0]["content"] == "RuntimeError"


def test_extract_evidence_warning():
    text = """
    Warning
    """
    evidence = extract_evidence(text)
    assert len(evidence) == 1
    assert evidence[0]["type"] == "WARNING"
    assert evidence[0]["content"] == "Warning"

def test_extract_evidence_numeric_anomaly():
    text = """
    loss = NaN
    loss = Inf
    """
    evidence = extract_evidence(text)
    assert len(evidence) == 2
    assert evidence[0]["type"] == "NUMERIC_ANOMALY"
    assert evidence[0]["content"] == "loss = NaN"
    assert evidence[1]["type"] == "NUMERIC_ANOMALY"
    assert evidence[1]["content"] == "loss = Inf"

def test_extract_evidence_does_not_match_inference():
    text = "Model inference completed successfully."

    evidence = extract_evidence(text)

    assert evidence == []

def test_extract_evidence_mixed():
    text = """
    This is a normal line.
    RuntimeError: Something went wrong.
    Warning: This might be an issue.
    loss = NaN
    loss = Inf
    """

    evidence = extract_evidence(text)

    assert len(evidence) == 4

    types = [item["type"] for item in evidence]

    assert types == [
        "ERROR",
        "WARNING",
        "NUMERIC_ANOMALY",
        "NUMERIC_ANOMALY",
    ]

def test_extract_evidence_multiple_labels_same_line():
    text = "WARNING: loss becomes NaN"

    evidence = extract_evidence(text)

    assert len(evidence) == 2

    types = {item["type"] for item in evidence}

    assert types == {
        "WARNING",
        "NUMERIC_ANOMALY",
    }

def test_extract_evidence_line_number():
    text = """normal line
another normal line
RuntimeError: CUDA out of memory"""

    evidence = extract_evidence(text)

    assert len(evidence) == 1
    assert evidence[0]["line_no"] == 3

def test_extract_multiline_traceback():
    text = """Traceback (most recent call last):
  File "train.py", line 10, in <module>
    train()
  File "train.py", line 5, in train
    loss.backward()
RuntimeError: CUDA out of memory"""

    evidence = extract_evidence(text)

    assert len(evidence) == 1
    assert evidence[0]["type"] == "ERROR"
    assert evidence[0]["line_no"] == 1
    assert "Traceback" in evidence[0]["content"]
    assert "loss.backward()" in evidence[0]["content"]
    assert "RuntimeError: CUDA out of memory" in evidence[0]["content"]

def test_traceback_is_not_duplicated():
    text = """Traceback (most recent call last):
  File "train.py", line 5, in train
    loss.backward()
RuntimeError: CUDA out of memory"""

    evidence = extract_evidence(text)

    assert len(evidence) == 1

def test_traceback_with_surrounding_evidence():
    text = """WARNING: training started
Traceback (most recent call last):
  File "train.py", line 5, in train
    loss.backward()
RuntimeError: CUDA out of memory
loss = NaN"""

    evidence = extract_evidence(text)

    types = [item["type"] for item in evidence]

    assert types == [
        "WARNING",
        "ERROR",
        "NUMERIC_ANOMALY",
    ]

def test_traceback_without_colon():
    text = """Traceback (most recent call last):
  File "train.py", line 10, in <module>
    train()
ValueError"""

    evidence = extract_evidence(text)

    assert len(evidence) == 1
    assert evidence[0]["type"] == "ERROR"
    assert evidence[0]["truncated"] is False
    assert "ValueError" in evidence[0]["content"]

def test_truncated_traceback_at_eof():
    text = """Traceback (most recent call last):
  File "train.py", line 10, in train
    loss.backward()"""

    evidence = extract_evidence(text)

    assert len(evidence) == 1
    assert evidence[0]["type"] == "ERROR"
    assert evidence[0]["truncated"] is True