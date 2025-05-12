from autogen.agents.experimental import DocAgent
import os
from autogen import ConversableAgent

DOC_PATH = "data/"
NUMBER = 5

def RAGAgent(doc_path: str, number: int, llm_config: dict) -> str:
    query = f"Create a list of the top {number} most important concepts. For each concept, explain it in a detailed manner and why it is important."
    llm_config = {'cache_seed': 42,
                    'temperature': 0.5,
                    'top_p': 0.05,
                    'config_list': [{'model': 'gpt-4o-mini',
                                    'api_key': os.getenv('OPENAI_API_KEY'),
                                    'api_type': 'openai'}],
                    'timeout': 1200}
    llm_config = {'cache_seed': 42,
                    'temperature': 0.5,
                    'top_p': 0.05,
                    'config_list': [{'model': 'gpt-4o-mini',
                                    'api_key': os.getenv('OPENAI_API_KEY'),
                                    'api_type': 'openai'}],
                    'timeout': 1200}
    doc_agent = DocAgent(llm_config=llm_config, collection_name='summarise')
    run_response = doc_agent.run(
        message = f"ingest all documents in {doc_path} and do {query}.",
        max_turns=1,
        
    )
    run_response.process()

    result = run_response.messages[1]['content']

    finalizer = ConversableAgent(
        name="Finalizer",
        llm_config=llm_config,
        system_message="Your job is to clean up the results from the DocAgent. Return the results in a list format, including as much detail as possible. Return ONLY the list with no other additional text.",
    )

    finalizer_response = finalizer.run(
        message=result,
        max_turns=1,
    )
    finalizer_response.process()
    final_result = finalizer_response.messages[1]['content']
    return final_result