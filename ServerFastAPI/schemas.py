from pydantic import BaseModel, field_validator
from typing import List, Optional

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.9
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False
    include_thinking: Optional[bool] = False

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if v < 0.0 or v > 2.0:
            raise ValueError('temperature must be between 0.0 and 2.0')
        return v

    @field_validator('top_p')
    @classmethod
    def validate_top_p(cls, v):
        if v <= 0.0 or v > 1.0:
            raise ValueError('top_p must be between 0.0 (exclusive) and 1.0')
        return v

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    include_thinking: Optional[bool] = False

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if v < 0.0 or v > 2.0:
            raise ValueError('temperature must be between 0.0 and 2.0')
        return v

    @field_validator('top_p')
    @classmethod
    def validate_top_p(cls, v):
        if v <= 0.0 or v > 1.0:
            raise ValueError('top_p must be between 0.0 (exclusive) and 1.0')
        return v
