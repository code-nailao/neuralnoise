from autogen import AssistantAgent
from typing import Any

from autogen import SwarmResult

from neuralnoise.studio.agents.context_manager import SharedContext


def create_editor_agent(
    system_msg: str,
    llm_config: dict,
) -> AssistantAgent:
    """Create and return an EditorAgent that reviews a section script and determines the next agent.

    Args:
        system_msg (str): The system message for the EditorAgent.
        llm_config (dict): The LLM configuration.

    Returns:
        AssistantAgent: The EditorAgent instance.
    """

    def provide_script_feedback(
        script: dict[str, Any],
        context_variables: dict[str, Any],
    ) -> SwarmResult:
        """Provide feedback on the script."""
        shared_state = SharedContext.model_validate(context_variables)

        return SwarmResult(
            values="Feedback provided",
            agent=None,
            context_variables=shared_state.model_dump(),
        )

    agent = AssistantAgent(
        name="EditorAgent",
        system_message=system_msg,
        llm_config=llm_config,
        functions=[provide_script_feedback],
    )

    return agent
