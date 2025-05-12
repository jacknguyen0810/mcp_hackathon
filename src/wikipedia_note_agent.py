import os
import shutil
from pathlib import Path
from pydantic import BaseModel, Field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from autogen import ConversableAgent, LLMConfig, UpdateSystemMessage
from autogen.mcp import create_toolkit
from autogen.agentchat import a_initiate_group_chat
from autogen.agentchat.group import (
    ContextVariables,
    AgentTarget,
    OnCondition,
    StringLLMCondition,
    TerminateTarget,
)
from autogen.agentchat.group.patterns import DefaultPattern


# Define la estructura de la nota de estudio
class StudyNoteResponse(BaseModel):
    title: str = Field(..., description="Note title")
    explanation: str = Field(..., description="Detail explanation")
    key_points: list[str] = Field(..., description="Key points to remember")

    def format(self) -> str:
        return f"# {self.title}\n\n{self.explanation}\n\n## Key points:\n" + \
               "\n".join(f"- {point}" for point in self.key_points)


# Clase principal del agente
class WikipediaNoteAgent:
    def __init__(self, study_context: str, openai_api_key: str):
        self.study_context = study_context
        self.openai_api_key = openai_api_key
        self.llm_model = "gpt-4o-mini"
        self.tool_storage_path = "mcp/wikipedia_articles"
        self.mcp_server_path = Path("mcp/mcp_wikipedia.py")

        # contexto compartido entre agentes
        self.workflow_context = ContextVariables(data={"study_context": self.study_context})

        # agentes
        self.note_writer = self._create_note_writer()
        self.mcp_agent = self._create_mcp_agent()

    def _create_note_writer(self) -> ConversableAgent:
        system_msg = """
                    You are an academic assistant.

                    Using the Wikipedia article summaries and the study context below, create a structured study note:
                    - A clear title
                    - A well-written explanation suitable for students
                    - A bullet-point list of key points

                    Study context:
                    {study_context}
                    """.strip()
        config = {
            "cache_seed": 42,
            "temperature": 0.7,
            "top_p": 0.9,
            "timeout": 1200,
            "config_list": [{
                "model": self.llm_model,
                "api_key": self.openai_api_key,
                "api_type": "openai",
                "response_format": StudyNoteResponse,
            }],
        }

        return ConversableAgent(
            name="note_writer",
            system_message=system_msg,
            llm_config=config,
            update_agent_state_before_reply=[UpdateSystemMessage(system_msg)]
        )

    def _create_mcp_agent(self) -> ConversableAgent:
        system_msg = f"""
                        You are a research assistant.

                        Your task is to search and download Wikipedia articles related to the following study context:

                        {self.study_context}

                        Once downloaded, summarize them and prepare the content for a study assistant.
                        """.strip()

        return ConversableAgent(
            name="mcp_agent",
            system_message=system_msg,
            llm_config=LLMConfig(
                model=self.llm_model,
                api_type="openai",
                api_key=self.openai_api_key,
                tool_choice="required"
            ),
        )

    def _clear_cache(self):
        cache_path = Path(".cache")
        if cache_path.exists():
            shutil.rmtree(cache_path)
            print(" .cache folder deleted.")

    async def run(self):
        # conexión con el servidor MCP
        server_params = StdioServerParameters(
            command="python",
            args=[str(self.mcp_server_path), "stdio", "--storage-path", self.tool_storage_path]
        )

        async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()

            toolkit = await create_toolkit(session=session)
            toolkit.register_for_llm(self.mcp_agent)
            toolkit.register_for_execution(self.mcp_agent)

            self.note_writer.handoffs.set_after_work(TerminateTarget())
            self.mcp_agent.handoffs.set_after_work(AgentTarget(self.note_writer))
            self.mcp_agent.handoffs.add_llm_conditions([
                OnCondition(
                    target=AgentTarget(self.note_writer),
                    condition=StringLLMCondition(prompt="The article has been downloaded and summarized."),
                ),
            ])

            for agent in [self.mcp_agent, self.note_writer]:
                agent.reset()

            self._clear_cache()

            pattern = DefaultPattern(
                agents=[self.mcp_agent, self.note_writer],
                initial_agent=self.mcp_agent,
                context_variables=self.workflow_context,
            )

            task_msg = "Use the study context to find Wikipedia articles and generate a student-friendly study note."

            await a_initiate_group_chat(
                pattern=pattern,
                messages=task_msg,
                max_rounds=20
            )

            final_note = None
            for sender, messages in self.note_writer.chat_messages.items():
                for msg in reversed(messages):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        final_note = msg["content"]
                        break
                if final_note:
                    break

            if final_note is None:
                raise RuntimeError("No assistant message found in note_writer output.")

            return final_note