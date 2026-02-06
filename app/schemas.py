from pydantic import BaseModel, Field
from typing import List, Optional

class PodcastRequest(BaseModel):
    topic: str = Field(..., description="The main topic of the podcast")
    tone: str = Field("Casual", description="The tone of the conversation (e.g., Casual, Formal, Debate)")


class DialogueTurn(BaseModel):
    Host: str = Field(..., description="The host's dialogue")
    Guest: str = Field(..., description="The guest's dialogue")

class PodcastResponse(BaseModel):
    status: str
    script: Optional[List[DialogueTurn]] = None
    original_script: Optional[str] = None
    language: str

class AgentResponse(BaseModel):
    role: str
    content: str

class DCSAnalysis(BaseModel):
    coherence_score: int = Field(..., description="1-10 score of coherence")
    coherence_reason: str = Field(..., description="Reason for the score")
    topic_drift: str = Field(..., description="'on_topic', 'slight_drift', or 'off_topic'")
    next_action: str = Field(..., description="Action for the host: 'standard_follow_up', 'clarify', 'steer_back', 'enthusiastic_interjection'")
    
    # New Metrics
    avg_coherence: float = Field(..., description="Running average of coherence scores (0-10)")
    topic_retention_pct: float = Field(..., description="Percentage of conversation remaining on main topic (0-100)")
    mean_opinion_score: float = Field(..., description="Estimated listener engagement score (1-5)")
    conversational_stability_index: float = Field(..., description="stability metric (0-1, where 1 is perfectly stable flow)")

class TranslationRequest(BaseModel):
    text: Optional[str] = Field(None, description="Single text to translate")
    script: Optional[List[DialogueTurn]] = Field(None, description="Full podcast script to translate")
    target_language: str = Field(..., description="Target language (Hindi, Kannada, Tamil, Telugu)")

class TranslationResponse(BaseModel):
    translated_text: Optional[str] = None
    translated_script: Optional[List[DialogueTurn]] = None
    script: Optional[List[DialogueTurn]] = None # Alias for translated_script to match TTSRequest
    original_text: Optional[str] = None
    original_script: Optional[List[DialogueTurn]] = None
    language: str

class TTSRequest(BaseModel):
    script: List[DialogueTurn] = Field(..., description="Full podcast script to convert to speech")
    language: str = Field(..., description="Target language for voice selection")

class TTSResponse(BaseModel):
    audio_file: str
    language: str
