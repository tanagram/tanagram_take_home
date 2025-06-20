
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class AppSettings:
    model: str = field(default='claude-3-7-sonnet-20250219')
    api_key: str = field(default=' ')
    api_base: Optional[str] = field(default='https://api.anthropic.com/api/v1')
    temperature: float = field(default=float(0.2))
    max_tokens: int = field(default=int(os.getenv("LLM_MAX_TOKENS", 6400)))



app_settings = AppSettings()