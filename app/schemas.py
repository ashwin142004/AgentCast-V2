from pydantic import BaseModel, Field
from typing import List, Optional

class PodcastRequest(BaseModel):
    topic: str = Field(..., description="The main topic of the podcast")
    tone: str = Field("Casual", description="The tone of the conversation (e.g., Casual, Formal, Debate)")
    language: str = Field("English", description="Target language (English, Hindi, Kannada, Tamil, Telugu)")

class PodcastResponse(BaseModel):
    status: str
    script: Optional[str] = None
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
