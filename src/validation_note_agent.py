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

class ConsistencyVerifierAgent:
    def __init__(self, openai_api_key: str, llm_model="gpt-4o-mini"):
        self.llm_config = {
            "cache_seed": 42,
            "temperature": 0.5,
            "top_p": 0.9,
            "timeout": 600,
            "config_list": [{
                "model": llm_model,
                "api_key": openai_api_key,
                "api_type": "openai",
            }],
        }

        self.agent = ConversableAgent(
            name="consistency_checker",
            system_message="""
                You are an expert reviewer. You are given two study notes.

                Your job is to:
                - Check for any inconsistencies between them.
                - If inconsistent, prefer the content from Note 1.
                - Merge both notes into one comprehensive version.
                - Be clear and student-friendly in your explanation.

                Return only the merged, verified note.
            """.strip(),
            llm_config=self.llm_config,
            silent=True
        )

    async def run(self, note_1: str, note_2: str) -> str:
        prompt = f"""
        Note 1:
        {note_1}

        Note 2:
        {note_2}

        Perform your review and return the merged, accurate study note.
        """

        self.agent.reset()
        reply = await self.agent.a_generate_reply(
            messages=[{"role": "user", "content": prompt}]
            )
        return reply