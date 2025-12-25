import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock heavy dependencies BEFORE importing app logic
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["indicnlp"] = MagicMock()
sys.modules["indicnlp.transliterate"] = MagicMock()
sys.modules["indicnlp.transliterate.unicode_transliterate"] = MagicMock()

# Mock environment variables before importing app
with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-api-key"}):
    from main import app
    from app.schemas import PodcastResponse, TranslationResponse

@pytest.fixture
def client():
    # raise_server_exceptions=False allows the tests to inspect 500 responses
    return TestClient(app, raise_server_exceptions=False)

@pytest.fixture
def mock_llm_chain():
    # Patch the LLM objects where they are used in agents
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value.content = "Mocked Response"
    
    with patch("app.agents.host.llm", mock_llm_instance), \
         patch("app.agents.guest.llm", mock_llm_instance), \
         patch("app.agents.judge.judge_llm") as mock_judge_llm:
         
         # Mock Judge
         mock_structured_llm = MagicMock()
         mock_judge_llm.with_structured_output.return_value = mock_structured_llm
         
         # The judge returns a Pydantic model usually, but we mocked .dict() access in tests
         mock_analysis_result = MagicMock()
         mock_analysis_result.coherence_score = 10
         mock_analysis_result.coherence_reason = "Good"
         mock_analysis_result.topic_drift = "on_topic"
         mock_analysis_result.next_action = "std_follow_up"
         mock_analysis_result.dict.return_value = {
             "coherence_score": 10,
             "coherence_reason": "Good",
             "topic_drift": "on_topic",
             "next_action": "std_follow_up"
         }
         
         mock_structured_llm.invoke.return_value = mock_analysis_result
         
         yield

@pytest.fixture
def mock_translator():
    with patch("app.agents.translator.get_translator") as mock_get_translator:
        mock_instance = MagicMock()
        mock_instance.translate.return_value = "Translated Text"
        mock_get_translator.return_value = mock_instance
        yield
