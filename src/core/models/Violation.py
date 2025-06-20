from pydantic import BaseModel
from typing import List

class Violation(BaseModel):
    """
    Represents a policy violation in code.
    Structure similar to Violation .proto file.
    """
    is_violation: bool              # Used because sometimes the LLM decides to say "It's not a violation"
    line_number_one_based: int
    content: str
    prompt: str
    explanation: str
    file_path: str

class ViolationList(BaseModel):
    """Represents a list of violations."""
    violations: List[Violation]