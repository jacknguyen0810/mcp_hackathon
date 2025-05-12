import os
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field, validator

try:
    from autogen import AssistantAgent, UserProxyAgent, LLMConfig
    AG2_AVAILABLE = True
except ImportError:
    AG2_AVAILABLE = False
    print("AG2 not available. Install with: pip install ag2[openai]")

# Define Pydantic models for structured output
class MultipleChoiceQuestion(BaseModel):
    id: int = Field(..., description="Unique identifier for the question")
    type: str = Field("multiple_choice", description="Type of question")
    question: str = Field(..., description="The question text")
    options: List[str] = Field(..., description="List of 4 options (A, B, C, D)")
    correct_answer: str = Field(..., description="The correct option (A, B, C, or D)")

    @validator('correct_answer')
    def validate_correct_answer(cls, value):
        if value not in ['A', 'B', 'C', 'D']:
            raise ValueError('correct_answer must be one of A, B, C, or D')
        return value
    
    @validator('options')
    def validate_options(cls, options):
        if len(options) != 4:
            raise ValueError('There must be exactly 4 options')
        return options

class LongAnswerQuestion(BaseModel):
    id: int = Field(..., description="Unique identifier for the question")
    type: str = Field("long_answer", description="Type of question")
    question: str = Field(..., description="The long-answer question text")

class QuestionItem(BaseModel):
    __root__: Union[MultipleChoiceQuestion, LongAnswerQuestion]

class QuestionsOutput(BaseModel):
    topic: str = Field(..., description="The original topic")
    questions: List[Union[MultipleChoiceQuestion, LongAnswerQuestion]] = Field(
        ..., description="List of questions"
    )

class AnswerItem(BaseModel):
    id: int = Field(..., description="ID of the question this answer belongs to")
    answer: str = Field(..., description="The answer or correct option for multiple choice")
    explanation: Optional[str] = Field(None, description="Detailed explanation of the answer")

class AnswersOutput(BaseModel):
    topic: str = Field(..., description="The original topic")
    answers: List[AnswerItem] = Field(..., description="List of answers")

# JSON schema representations for prompting
QUESTION_SCHEMA = QuestionsOutput.schema()
ANSWER_SCHEMA = AnswersOutput.schema()

class QuestionGeneratorAgent:
    """
    An agent that generates educational questions based on provided prompts.
    Can create multiple choice and long-answer questions suitable for university students.
    """
    
    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        save_dir: str = "generated_questions"
    ):
        """
        Initialize the Question Generator Agent.
        
        Args:
            llm_config: Configuration for the language model
            verbose: Whether to print detailed information
            save_dir: Directory to save generated questions and answers
        """
        if not AG2_AVAILABLE:
            raise ImportError("AG2 is required. Install with: pip install ag2[openai]")
            
        self.verbose = verbose
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Configure LLM settings
        if llm_config is None:
            # Default configuration
            llm_config = LLMConfig(
                model="gpt-4-turbo",
                config_list=[{"model": "gpt-4-turbo", "api_key": os.environ.get("OPENAI_API_KEY")}]
            )
        
        # System message for the assistant that generates questions
        question_system_message = f"""
        You are an expert educational content creator for university students.
        
        Your task is to generate educational questions based on the provided topic or prompt.
        
        INSTRUCTIONS:
        1. Generate a mix of multiple-choice and long-answer questions
        2. Make questions appropriate for university-level students
        3. Create challenging but fair questions that test understanding, not just recall
        4. For multiple-choice questions, provide 4 options with one correct answer
        5. Ensure all questions are clear, concise, and unambiguous
        6. Create questions that encourage critical thinking
        
        Your response should follow this JSON schema:
        ```json
        {json.dumps(QUESTION_SCHEMA, indent=2)}
        ```
        
        Make sure to ONLY respond with valid JSON that matches this schema.
        For multiple-choice questions, always use A, B, C, D as the correct_answer value.
        """
        
        # System message for the assistant that generates answers
        answer_system_message = f"""
        You are an expert educational answer key creator for university students.
        
        Your task is to provide detailed, educational answers to the given questions.
        
        INSTRUCTIONS:
        1. Provide clear and accurate answers for each question
        2. For multiple-choice questions, explain why the correct answer is right and why others are wrong
        3. For long-answer questions, provide a comprehensive model answer that would receive full marks
        4. Include relevant examples, theories, or formulas where appropriate
        5. Make your explanations educational and helpful for university students

        Your response should follow this JSON schema:
        ```json
        {json.dumps(ANSWER_SCHEMA, indent=2)}
        ```
        
        Make sure to ONLY respond with valid JSON that matches this schema.
        """
        
        # Create the assistants with AG2
        self.question_assistant = AssistantAgent(
            name="QuestionGenerator",
            system_message=question_system_message,
            llm_config=llm_config
        )
        
        self.answer_assistant = AssistantAgent(
            name="AnswerGenerator",
            system_message=answer_system_message,
            llm_config=llm_config
        )
        
        # Create a user proxy that will interact with the assistants
        # This proxy doesn't take human input but terminates after one response
        self.user_proxy = UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            is_termination_msg=lambda x: True,  # Single turn conversation
            code_execution_config=False  # Disable code execution
        )
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text, handling various formats."""
        if not text:
            raise ValueError("Empty response")
            
        # If the text is already valid JSON, return it directly
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        # Try to extract JSON from markdown code blocks
        import re
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(json_pattern, text)
        
        if matches:
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
                    
        # If no valid JSON found in code blocks, try to find JSON object in the text
        try:
            # Find the first { and last }
            start = text.find('{')
            end = text.rfind('}')
            
            if start >= 0 and end > start:
                json_str = text[start:end+1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
        raise ValueError("Could not extract valid JSON from the response")
    
    def generate_questions(self, topic: str, num_questions: int = 5) -> Dict[str, Any]:
        """
        Generate questions based on the given topic.
        
        Args:
            topic: The prompt or topic to generate questions about
            num_questions: Number of questions to generate
            
        Returns:
            Dictionary containing generated questions
        """
        prompt = f"""
        Topic: {topic}
        
        Please generate {num_questions} questions about this topic for university students.
        Create a mix of multiple-choice and long-answer questions.
        """
        
        if self.verbose:
            print(f"Generating questions for topic: {topic}")
        
        # Get the response from the question assistant
        response = self.user_proxy.initiate_chat(
            self.question_assistant,
            message=prompt
        )
        
        # Extract the last response from the assistant
        messages = response.chat_history
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                content = msg["content"]
                # Try to extract JSON
                try:
                    # Extract JSON from response
                    json_data = self._extract_json(content)
                    
                    # Validate with Pydantic
                    questions_data = QuestionsOutput.parse_obj(json_data)
                    
                    # Convert back to dict for consistency with the rest of the code
                    return questions_data.dict()
                except Exception as e:
                    if self.verbose:
                        print(f"Error validating questions: {str(e)}")
                    return {"error": f"Invalid question format: {str(e)}"}
        
        return {"error": "Failed to get a response from the assistant"}
    
    def generate_answers(self, questions_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate answers for the previously generated questions.
        
        Args:
            questions_data: Dictionary containing the questions
            
        Returns:
            Dictionary containing answers for the questions
        """
        if "error" in questions_data:
            return questions_data
            
        # Convert questions to a JSON string for the prompt
        questions_json = json.dumps(questions_data, indent=2)
        
        prompt = f"""
        Please provide detailed answers for these questions:
        
        {questions_json}
        """
        
        if self.verbose:
            print("Generating answers for the questions")
        
        # Get the response from the answer assistant
        response = self.user_proxy.initiate_chat(
            self.answer_assistant,
            message=prompt
        )
        
        # Extract the last response from the assistant
        messages = response.chat_history
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                content = msg["content"]
                # Try to extract JSON
                try:
                    # Extract JSON from response
                    json_data = self._extract_json(content)
                    
                    # Validate with Pydantic
                    answers_data = AnswersOutput.parse_obj(json_data)
                    
                    # Convert back to dict for consistency with the rest of the code
                    return answers_data.dict()
                except Exception as e:
                    if self.verbose:
                        print(f"Error validating answers: {str(e)}")
                    return {"error": f"Invalid answer format: {str(e)}"}
        
        return {"error": "Failed to get a response from the assistant"}
    
    def save_to_files(self, questions_data: Dict[str, Any], answers_data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Save questions and answers to separate files.
        
        Args:
            questions_data: Dictionary containing the questions
            answers_data: Dictionary containing the answers
            
        Returns:
            Tuple of file paths (questions_file, answers_file)
        """
        if "error" in questions_data or "error" in answers_data:
            return "", ""
            
        # Create a sanitized filename from the topic
        topic = questions_data.get("topic", "unknown_topic")
        safe_topic = "".join(c if c.isalnum() else "_" for c in topic).lower()
        timestamp = str(int(time.time()))
        
        # Save questions
        questions_file = os.path.join(self.save_dir, f"{safe_topic}_questions_{timestamp}.json")
        with open(questions_file, 'w', encoding='utf-8') as f:
            json.dump(questions_data, f, indent=2)
            
        # Save answers
        answers_file = os.path.join(self.save_dir, f"{safe_topic}_answers_{timestamp}.json")
        with open(answers_file, 'w', encoding='utf-8') as f:
            json.dump(answers_data, f, indent=2)
            
        if self.verbose:
            print(f"Questions saved to: {questions_file}")
            print(f"Answers saved to: {answers_file}")
            
        return questions_file, answers_file
        
    def generate(self, topic: str, num_questions: int = 5, save: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate both questions and answers for a topic.
        
        Args:
            topic: The prompt or topic to generate questions about
            num_questions: Number of questions to generate
            save: Whether to save the output to files
            
        Returns:
            Tuple of (questions_data, answers_data)
        """
        # Generate questions
        questions_data = self.generate_questions(topic, num_questions)
        if "error" in questions_data:
            return questions_data, {"error": "Failed to generate questions"}
            
        # Generate answers
        answers_data = self.generate_answers(questions_data)
        
        # Save to files if requested
        if save:
            self.save_to_files(questions_data, answers_data)
            
        return questions_data, answers_data 