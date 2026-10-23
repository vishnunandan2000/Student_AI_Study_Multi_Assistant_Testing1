"""
Pydantic Schemas for GenAI Video Intelligence Studio.
Provides structured data validation for:
- Video comments, sentiment classification, audience feedback
- Calculated engagement metrics
- Edit Decision List (EDL) items
- Study notes chapters
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class CommentRecord(BaseModel):
    comment_id: str
    text: str
    author: str = "Anonymous"
    timestamp: Optional[str] = None
    like_count: int = 0
    sentiment: Literal["Positive", "Negative", "Neutral", "Unclassified"] = "Unclassified"
    explanation: str = ""
    topic: str = "General"
    actionable_feedback: bool = False
    uncertainty: bool = False


class SentimentSummary(BaseModel):
    positive_count: int = 0
    positive_percent: float = 0.0
    negative_count: int = 0
    negative_percent: float = 0.0
    neutral_count: int = 0
    neutral_percent: float = 0.0
    unclassified_count: int = 0
    total_classified: int = 0
    common_concerns: List[str] = Field(default_factory=list)
    repeated_requests: List[str] = Field(default_factory=list)
    frequently_praised: List[str] = Field(default_factory=list)



class EditDecisionItem(BaseModel):
    start_seconds: float
    end_seconds: float
    operation: str
    reason: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    associated_animation_or_caption: Optional[str] = None
    validation_status: str = "valid"


class StudyNoteChapter(BaseModel):
    chapter_num: int
    title: str
    timestamp_str: str = "00:00"
    definitions: List[str] = Field(default_factory=list)
    formulas_or_code: List[str] = Field(default_factory=list)
    explanations: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    key_takeaways: List[str] = Field(default_factory=list)
    revision_questions: List[str] = Field(default_factory=list)
