import os
import json
import time
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field, RootModel, field_validator
from dotenv import load_dotenv
load_dotenv()

try:
    from autogen import AssistantAgent, UserProxyAgent, LLMConfig, ConversableAgent
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

    @field_validator('correct_answer')
    @classmethod
    def validate_correct_answer(cls, value):
        if value not in ['A', 'B', 'C', 'D']:
            raise ValueError('correct_answer must be one of A, B, C, or D')
        return value
    
    @field_validator('options')
    @classmethod
    def validate_options(cls, options):
        if len(options) != 4:
            raise ValueError('There must be exactly 4 options')
        return options

# Updated for multiple choice questions only
class QuestionsOutput(BaseModel):
    topic: str = Field(..., description="The original topic")
    questions: List[MultipleChoiceQuestion] = Field(
        ..., description="List of multiple choice questions"
    )

class AnswerItem(BaseModel):
    id: int = Field(..., description="ID of the question this answer belongs to")
    answer: str = Field(..., description="The correct option for multiple choice")
    explanation: Optional[str] = Field(None, description="Detailed explanation of the answer")

class AnswersOutput(BaseModel):
    topic: str = Field(..., description="The original topic")
    answers: List[AnswerItem] = Field(..., description="List of answers")

# JSON schema representations for prompting
QUESTION_SCHEMA = QuestionsOutput.model_json_schema()
ANSWER_SCHEMA = AnswersOutput.model_json_schema()

class QuestionGeneratorAgent:
    """
    An agent that generates educational multiple choice questions based on provided prompts.
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
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
                
            # Use the new format for LLMConfig without response_format
            llm_config = {
                "config_list": [{
                    "model": "gpt-4o-mini",
                    "api_key": api_key
                }]
            }
            
        # System message for the question generator
        question_system_message = f"""
        You are an expert educational content creator for university students.
        
        Your task is to generate MULTIPLE CHOICE QUESTIONS ONLY based on the provided topic or prompt.
        
        INSTRUCTIONS:
        1. Generate ONLY multiple-choice questions (no long-answer questions)
        2. Make questions appropriate for university-level students
        3. Create challenging but fair questions that test understanding, not just recall
        4. For each question, provide 4 options labeled A, B, C, and D
        5. Ensure all questions are clear, concise, and unambiguous
        6. Create questions that encourage critical thinking
        
        Your response should follow this JSON schema:
        ```json
        {json.dumps(QUESTION_SCHEMA, indent=2)}
        ```
        
        Make sure to ONLY respond with valid JSON that matches this schema.
        The correct_answer field must be one of: A, B, C, or D.
        """
        
        # System message for the answer generator
        answer_system_message = f"""
        You are an expert educational answer key creator for university students.
        
        Your task is to provide detailed, educational answers to the given multiple choice questions.
        
        INSTRUCTIONS:
        1. Provide clear and accurate answers for each question
        2. Explain why the correct answer is right and why others are wrong
        3. Include relevant examples, theories, or formulas where appropriate
        4. Make your explanations educational and helpful for university students

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
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(json_pattern, text)
        
        if matches:
            for match in matches:
                try:
                    cleaned_match = match.strip()
                    if self.verbose:
                        print(f"Trying to parse JSON: {cleaned_match[:100]}...")
                    return json.loads(cleaned_match)
                except json.JSONDecodeError:
                    continue
                    
        # If no valid JSON found in code blocks, try to find JSON object in the text
        try:
            # Find the first { and last }
            start = text.find('{')
            end = text.rfind('}')
            
            if start >= 0 and end > start:
                json_str = text[start:end+1]
                if self.verbose:
                    print(f"Trying to parse JSON from raw text: {json_str[:100]}...")
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
        if self.verbose:
            print("JSON extraction failed. Raw response:")
            print(text)
            
        raise ValueError("Could not extract valid JSON from the response")
    
    def generate_questions(self, topic: str, num_questions: int = 5) -> Dict[str, Any]:
        """
        Generate multiple choice questions based on the given topic.
        
        Args:
            topic: The prompt or topic to generate questions about
            num_questions: Number of questions to generate
            
        Returns:
            Dictionary containing generated questions
        """
        prompt = f"""
        Topic: {topic}
        
        Please generate {num_questions} multiple choice questions about this topic for university students.
        Each question must have 4 options (A, B, C, D) with one correct answer.
        
        IMPORTANT: Respond with ONLY a valid JSON object following the schema provided.
        """
        
        if self.verbose:
            print(f"Generating multiple choice questions for topic: {topic}")
        
        try:
            # Get the response from the question assistant
            response = self.user_proxy.initiate_chat(
                self.question_assistant,
                message=prompt
            )
            
            # Show the complete chat history for debugging
            if self.verbose:
                print("Complete chat history:")
                for i, msg in enumerate(response.chat_history):
                    role = msg.get("role", "unknown")
                    content_preview = (msg.get("content", "")[:50] + "...") if len(msg.get("content", "")) > 50 else msg.get("content", "")
                    print(f"Message {i} - {role}: {content_preview}")
            
            # Simplest: parse the last message content
            last_msg = response.chat_history[-1].get("content", "")
            if self.verbose:
                print(f"Parsing JSON from last message: {last_msg[:100]}...")
            start = last_msg.find("{")
            end = last_msg.rfind("}")
            if start < 0 or end < 0:
                return {"error": "No JSON object found in assistant response"}
            json_str = last_msg[start:end+1]
            try:
                json_data = json.loads(json_str)
                # Ensure type field
                for q in json_data.get("questions", []):
                    q.setdefault("type", "multiple_choice")
                questions_data = QuestionsOutput.model_validate(json_data)
                return questions_data.model_dump()
            except Exception as e:
                if self.verbose:
                    print(f"Error parsing JSON: {e}")
                return {"error": f"Invalid question format: {str(e)}"}
                
        except Exception as e:
            if self.verbose:
                print(f"Exception in generate_questions: {str(e)}")
            return {"error": f"Exception in generate_questions: {str(e)}"}
    
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
        Please provide detailed answers for these multiple choice questions:
        
        {questions_json}
        
        IMPORTANT: Respond with ONLY a valid JSON object containing the answers.
        Each answer should include the question id, the letter of the correct answer, and an explanation.
        """
        
        if self.verbose:
            print("Generating answers for the questions")
        
        try:
            # Get the response from the answer assistant
            response = self.user_proxy.initiate_chat(
                self.answer_assistant,
                message=prompt
            )
            
            # Show the complete chat history for debugging
            if self.verbose:
                print("Complete answer chat history:")
                for i, msg in enumerate(response.chat_history):
                    role = msg.get("role", "unknown")
                    content_preview = (msg.get("content", "")[:50] + "...") if len(msg.get("content", "")) > 50 else msg.get("content", "")
                    print(f"Answer message {i} - {role}: {content_preview}")
            
            # Parse JSON from last message only
            last_msg = response.chat_history[-1].get("content", "")
            if self.verbose:
                print(f"Parsing answers JSON from last message: {last_msg[:100]}...")
            start = last_msg.find("{")
            end = last_msg.rfind("}")
            if start < 0 or end < 0:
                return {"error": "No JSON object found in answer response"}
            json_str = last_msg[start:end+1]
            try:
                json_data = json.loads(json_str)
                answers_data = AnswersOutput.model_validate(json_data)
                return answers_data.model_dump()
            except Exception as e:
                if self.verbose:
                    print(f"Error parsing answers JSON: {e}")
                # Fallback minimal answers
                minimal = []
                for q in questions_data.get("questions", []):
                    minimal.append({"id": q["id"], "answer": q["correct_answer"], "explanation": f"The correct answer is {q['correct_answer']}."})
                return {"topic": questions_data.get("topic", ""), "answers": minimal}
                
        except Exception as e:
            if self.verbose:
                print(f"Exception in generate_answers: {str(e)}")
            return {"error": f"Exception in generate_answers: {str(e)}"}
    
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
            if self.verbose:
                print(f"Failed to generate questions: {questions_data['error']}")
            return questions_data, {"error": "Failed to generate questions"}
            
        # If we have valid questions, try to generate answers
        if not questions_data.get("questions"):
            if self.verbose:
                print("No questions were generated.")
            return questions_data, {"error": "No questions generated"}
            
        try:
            # Generate answers
            answers_data = self.generate_answers(questions_data)
            
            # If answer generation failed but we have questions, create minimal placeholder answers
            if "error" in answers_data and questions_data.get("questions"):
                if self.verbose:
                    print(f"Warning: Failed to generate answers: {answers_data['error']}")
                    print("Creating minimal placeholder answers...")
                
                # Create minimal answers
                minimal_answers = []
                for q in questions_data["questions"]:
                    minimal_answers.append({
                        "id": q["id"],
                        "answer": q["correct_answer"],
                        "explanation": f"The correct answer is {q['correct_answer']}."
                    })
                
                answers_data = {
                    "topic": questions_data.get("topic", topic),
                    "answers": minimal_answers
                }
                
                if self.verbose:
                    print(f"Created {len(minimal_answers)} placeholder answers.")
            
            # Save to files if requested
            if save and "error" not in answers_data:
                self.save_to_files(questions_data, answers_data)
                
            return questions_data, answers_data
            
        except Exception as e:
            if self.verbose:
                print(f"Exception in generate: {str(e)}")
            return questions_data, {"error": f"Exception generating answers: {str(e)}"} 