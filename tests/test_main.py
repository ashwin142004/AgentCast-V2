from fastapi.testclient import TestClient
from unittest.mock import patch

def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "AgentCast V2"}

def test_generate_podcast_success(client, mock_llm_chain, mock_translator):
    payload = {
        "topic": "AI Agents",
        "tone": "Casual",
        "language": "Hindi" # Triggers translation
    }
    response = client.post("/generate-podcast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["script"] == "Translated Text"
    assert data["language"] == "Hindi"

def test_generate_podcast_validation_error(client):
    payload = {
        "tone": "Casual" 
        # Missing 'topic'
    }
    response = client.post("/generate-podcast", json=payload)
    assert response.status_code == 422

def test_translate_text_success(client, mock_translator):
    payload = {
        "text": "Hello World",
        "target_language": "Tamil"
    }
    response = client.post("/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["translated_text"] == "Translated Text"
    assert data["language"] == "Tamil"

def test_translate_text_error(client):
    # Mocking an exception inside translate_script
    with patch("app.agents.translator.translate_script", side_effect=Exception("Model Failure")):
        payload = {
            "text": "Hello World",
            "target_language": "Tamil"
        }
        response = client.post("/translate", json=payload)
        
        # Should be caught by global exception handler
        assert response.status_code == 500
        assert "Model Failure" in response.json()["detail"]
