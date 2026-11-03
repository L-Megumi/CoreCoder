import pytest
from corecoder.llm import ScriptedLLM, LLMResponse
from corecoder.tools import ALL_TOOLS
from researchcoder.runtime.isolation import isolated_agent, WorkspaceLease


def test_two_agents_have_separate_tools_parents_context_and_usage():
    parent = next(t for t in ALL_TOOLS if t.name == "agent")._parent_agent
    first = isolated_agent(ScriptedLLM([LLMResponse(content="first reply")]), allow_agent=True)
    second = isolated_agent(ScriptedLLM([LLMResponse(content="second reply")]), allow_agent=True)
    assert all(a is not b for a, b in zip(first.tools, second.tools))
    assert next(t for t in first.tools if t.name == "agent")._parent_agent is first
    assert next(t for t in second.tools if t.name == "agent")._parent_agent is second
    assert next(t for t in ALL_TOOLS if t.name == "agent")._parent_agent is parent
    first.context.max_tokens = 5000
    assert second.context.max_tokens != 5000
    assert first.chat("a") == "first reply"
    assert second.messages == [] and second.llm.total_completion_tokens == 0
    assert second.chat("b") == "second reply"
    assert first.messages[0]["content"] == "a" and second.messages[0]["content"] == "b"
    assert all(t.name != "agent" for t in isolated_agent(ScriptedLLM([])).tools)


def test_workspace_single_writer_lease_is_released_after_error(tmp_path):
    with WorkspaceLease(tmp_path):
        with pytest.raises(RuntimeError):
            with WorkspaceLease(tmp_path):
                pass
    with WorkspaceLease(tmp_path):
        pass
