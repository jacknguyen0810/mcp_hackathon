import asyncio
from wikipedia_note_agent import WikipediaNoteAgent
import nest_asyncio
import os

nest_asyncio.apply()

study_context = "Diffusion models are a type of machine learning models"
api_key = os.getenv("OPENAI_API_KEY")


agent = WikipediaNoteAgent(study_context=study_context, openai_api_key=api_key)
note = asyncio.run(agent.run())
print(note)