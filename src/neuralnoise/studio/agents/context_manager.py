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
    sections: list[dict[str, Any]] = Field(
        default_factory=list, description="List of content sections with their metadata"
    )
    section_scripts: dict[int, dict[str, Any]] = Field(
        default_factory=dict,
        description="Mapping of section indices to their associated scripts",
    )
    execution_plans: str = Field(
        default="",
        description="Execution plans for the complete podcast, specifying all required sections",
    )
    current_section_index: int = Field(
        default=1, description="Index of the currently active section"
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

    def update_content(self, content: str) -> None:
        """Update the raw content.

        Args:
            content: The new content to store
        """
        self.content = content

    def update_content_analysis(self, analysis: dict[str, Any]) -> None:
        """Update the content analysis results.

        Args:
            analysis: The analysis results to store
        """
        self.content_analysis = analysis

    def add_sections(self, sections: list[dict[str, Any]]) -> None:
        """Add new sections to the state.

        Args:
            sections: List of section data to append
        """
        self.sections.extend(sections)
        self.current_section_index = len(self.sections) - 1

    def update_section_script(self, script: dict[str, Any]) -> None:
        """Update the script for the current section.

        Args:
            script: The script data to associate with the current section
        """
        self.section_scripts[self.current_section_index] = script

    def get_current_section(self) -> dict[str, Any]:
        """Get the currently active section.

        Returns:
            The current section data or an empty dict if no sections exist
        """
        if self.sections:
            return self.sections[self.current_section_index]
        return {}

    def add_error(self, error_message: str) -> None:
        """Add an error message to the list of errors.

        Args:
            error_message: The error message to add
        """
        self.errors.append(error_message)

    def add_warning(self, warning_message: str) -> None:
        """Add a warning message to the list of warnings.

        Args:
            warning_message: The warning message to add
        """
        self.warnings.append(warning_message)

    def get_last_error(self) -> str | None:
        """Get the most recent error message.

        Returns:
            The last error message or None if no errors exist
        """
        return self.errors[-1] if self.errors else None

    def get_all_errors(self) -> list[str]:
        """Get all error messages.

        Returns:
            List of all error messages
        """
        return self.errors

    def has_errors(self) -> bool:
        """Check if any errors have been recorded.

        Returns:
            True if errors exist, False otherwise
        """
        return len(self.errors) > 0
