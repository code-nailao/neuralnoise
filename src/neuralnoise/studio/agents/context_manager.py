from typing import Any
from pydantic import BaseModel, Field


class SharedContext(BaseModel):
    """Manages shared state for content processing and section management."""

    content: str | None = Field(
        default=None, description="The raw content being processed"
    )
    content_analysis: dict[str, Any] | None = Field(
        default=None, description="Analysis results of the processed content"
    )
    section_scripts: dict[int, dict[str, Any]] = Field(
        default_factory=dict,
        description="Mapping of section indices to their associated scripts",
    )
    section_feedbacks: dict[int, list[str]] = Field(
        default_factory=dict,
        description="Mapping of section indices to their associated feedback",
    )
    execution_plans: str = Field(
        default="",
        description="Execution plans for the complete podcast, specifying all required sections",
    )
    current_section_index: int = Field(
        default=0, description="Index of the currently active section"
    )
    is_complete: bool = Field(
        default=False, description="Flag indicating if processing is complete"
    )
    errors: list[str] = Field(
        default_factory=list, description="List of errors encountered during processing"
    )
    warnings: list[str] = Field(
        default_factory=list, description="List of warnings generated during processing"
    )
