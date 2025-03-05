"""
AgentsManager module: Centralizes instantiation and handoff registration for swarm agents.
It also defines a final workflow to run the swarm chat and extract the final results.
"""

from typing import Sequence, Tuple

from autogen import (
    AfterWorkOption,
    AssistantAgent,
    ChatResult,
    ConversableAgent,
    UserProxyAgent,
    initiate_swarm_chat,
)

from neuralnoise.models import ContentAnalysis, PodcastScript
from neuralnoise.prompt_manager import PromptManager, PromptType
from neuralnoise.studio.agents.content_analyzer_agent import (
    create_content_analyzer_agent,
)
from neuralnoise.studio.agents.context_manager import SharedContext
from neuralnoise.studio.agents.editor_agent import create_editor_agent
from neuralnoise.studio.agents.planner_agent import create_planner_agent
from neuralnoise.studio.agents.script_generator_agent import (
    create_script_generator_agent,
)


class AgentsManager:
    def __init__(
        self,
        llm_config: dict,
        language: str,
    ) -> None:
        """
        Initialize the AgentsManager with required configuration parameters,
        instantiate all agents, and register handoffs.

        Args:
            llm_config (dict): LLM configuration parameters.
            language (str): Language identifier for prompts.
        """
        self.language: str = language
        self.llm_config: dict = llm_config
        self.agents: dict[str, AssistantAgent] = {}

        self.prompt_manager = PromptManager(language=language)

        self.agents["PlannerAgent"] = create_planner_agent(
            system_msg=self.prompt_manager.get_prompt(PromptType.PLANNER),
            llm_config=llm_config,
        )

        content_analyzer_llm_config = llm_config.copy()
        content_analyzer_llm_config["response_format"] = ContentAnalysis
        self.agents["ContentAnalyzerAgent"] = create_content_analyzer_agent(
            system_msg=self.prompt_manager.get_prompt(PromptType.CONTENT_ANALYZER),
            llm_config=content_analyzer_llm_config,  # Use the modified config with response_format
            language=language,
        )

        script_generator_llm_config = llm_config.copy()
        script_generator_llm_config["response_format"] = PodcastScript
        self.agents["ScriptGeneratorAgent"] = create_script_generator_agent(
            system_msg=self.prompt_manager.get_prompt(PromptType.SCRIPT_GENERATOR),
            llm_config=script_generator_llm_config,
        )

        self.agents["EditorAgent"] = create_editor_agent(
            system_msg=self.prompt_manager.get_prompt(PromptType.EDITOR),
            llm_config=llm_config,
        )

    def run_swarm_chat(
        self, initial_message: str
    ) -> Tuple[ChatResult, SharedContext, ConversableAgent]:
        """
        Set up the shared state and user proxy, and initiate the swarm chat flow.

        Args:
            initial_message (str): The initial message or prompt for the swarm chat.

        Returns:
            Tuple[ChatResult, SharedState, ConversableAgent]: A tuple containing the conversation log,
            the final shared state, and the last agent that handled the conversation.
        """
        # Instantiate shared state.
        shared_state = SharedContext()

        # Initialize content from initial message if not already set
        if not shared_state.content and initial_message:
            shared_state.content = initial_message

        # Prepare list of agents.
        swarm_agents: Sequence[ConversableAgent] = list(self.agents.values())

        user_proxy = UserProxyAgent(
            name="UserProxyAgent",
            human_input_mode="NEVER",
            code_execution_config=False,
        )

        # Initiate swarm chat starting with the ContentAnalyzerAgent.
        chat_result, final_context, last_agent = initiate_swarm_chat(
            initial_agent=self.agents["ContentAnalyzerAgent"],
            agents=swarm_agents,
            messages=initial_message,
            context_variables=shared_state.model_dump(),
            user_agent=user_proxy,
            swarm_manager_args={
                "human_input_mode": "NEVER",
                "system_message": self.prompt_manager.get_prompt(PromptType.MANAGER),
                "llm_config": self.llm_config,
            },
            after_work=AfterWorkOption.SWARM_MANAGER,
            max_rounds=200,
        )

        # Update shared state with final context.
        shared_state = shared_state.model_validate(final_context)

        return chat_result, shared_state, last_agent
