import os
import json
import sys
import logging
from typing import Dict, List, Any, Optional, Tuple
import requests
from datetime import datetime
from question_generator import QuestionGeneratorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='auto_anki_mcp.log'  # Log to file to avoid polluting stdout
)
logger = logging.getLogger(__name__)

class AnkiCardGenerator:
    """
    A class that converts questions and answers into Anki card format
    """
    
    @staticmethod
    def format_multiple_choice(question_data: Dict[str, Any]) -> Tuple[str, str]:
        """Format a multiple choice question for Anki"""
        question = question_data["question"]
        options = question_data["options"]
        
        # Format the front of the card
        front = f"{question}\n\n"
        for i, option in enumerate(options):
            front += f"{chr(65+i)}. {option}\n"
        
        # Format the back of the card
        correct_letter = question_data["correct_answer"][0]  # Get the letter (A, B, C, D)
        correct_index = ord(correct_letter) - 65  # Convert to index (0, 1, 2, 3)
        correct_option = options[correct_index]
        
        back = f"Answer: {correct_letter}. {correct_option}\n\n"
        back += f"Explanation: {question_data.get('explanation', '')}"
        
        return front, back
    
    @staticmethod
    def format_long_answer(question_data: Dict[str, Any], answer_data: Dict[str, Any]) -> Tuple[str, str]:
        """Format a long answer question for Anki"""
        question = question_data["question"]
        answer = answer_data["answer"]
        explanation = answer_data.get("explanation", "")
        
        front = question
        
        # Format the back of the card
        if explanation:
            back = f"{answer}\n\nAdditional notes:\n{explanation}"
        else:
            back = answer
            
        return front, back
    
    @staticmethod
    def convert_to_anki_cards(questions_data: Dict[str, Any], answers_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert questions and answers into Anki card format
        
        Returns:
            List of dicts with 'front' and 'back' fields
        """
        cards = []
        
        # Create a mapping of question ID to answer for easier lookup
        answer_map = {answer["id"]: answer for answer in answers_data["answers"]}
        
        for question in questions_data["questions"]:
            q_id = question["id"]
            q_type = question["type"]
            
            if q_id in answer_map:
                answer_data = answer_map[q_id]
                
                if q_type == "multiple_choice":
                    # Add explanation from answer data to question data for formatting
                    question["explanation"] = answer_data.get("explanation", "")
                    front, back = AnkiCardGenerator.format_multiple_choice(question)
                elif q_type == "long_answer":
                    front, back = AnkiCardGenerator.format_long_answer(question, answer_data)
                else:
                    logger.warning(f"Unsupported question type: {q_type}")
                    continue
                
                cards.append({
                    "front": front,
                    "back": back
                })
        
        return cards

class AnkiConnector:
    """
    Handles communication with Anki via AnkiConnect
    """
    def __init__(self, anki_connect_url: str = "http://localhost:8765"):
        self.anki_connect_url = anki_connect_url
    
    def check_connection(self) -> bool:
        """Check if AnkiConnect is available"""
        try:
            response = requests.post(
                self.anki_connect_url,
                json={
                    "action": "version",
                    "version": 6
                }
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def create_deck(self, deck_name: str) -> bool:
        """Create an Anki deck if it doesn't exist"""
        try:
            response = requests.post(
                self.anki_connect_url,
                json={
                    "action": "createDeck",
                    "version": 6,
                    "params": {
                        "deck": deck_name
                    }
                }
            )
            result = response.json()
            return "error" not in result
        except requests.RequestException as e:
            logger.error(f"Failed to create Anki deck: {str(e)}")
            return False
    
    def add_card(self, front: str, back: str, deck_name: str = "Default") -> int:
        """
        Add a card to Anki
        
        Returns:
            Card ID if successful, 0 otherwise
        """
        try:
            # Make sure deck exists
            self.create_deck(deck_name)
            
            # Add the note (card)
            response = requests.post(
                self.anki_connect_url,
                json={
                    "action": "addNote",
                    "version": 6,
                    "params": {
                        "note": {
                            "deckName": deck_name,
                            "modelName": "Basic",
                            "fields": {
                                "Front": front,
                                "Back": back
                            },
                            "tags": ["auto_anki", "mcp"]
                        }
                    }
                }
            )
            result = response.json()
            if "error" not in result:
                return result["result"]  # This is the noteId
            else:
                logger.warning(f"Failed to add card: {result.get('error')}")
                return 0
        except requests.RequestException as e:
            logger.error(f"Request error adding card: {str(e)}")
            return 0
    
    def update_card(self, card_id: int, ease: int) -> bool:
        """
        Mark a card as answered with the given ease
        
        Args:
            card_id: The card ID
            ease: 1 (Again), 2 (Hard), 3 (Good), or 4 (Easy)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.post(
                self.anki_connect_url,
                json={
                    "action": "answerCard",
                    "version": 6,
                    "params": {
                        "cardId": card_id,
                        "ease": ease
                    }
                }
            )
            result = response.json()
            return "error" not in result
        except requests.RequestException as e:
            logger.error(f"Request error updating card: {str(e)}")
            return False
    
    def get_cards(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get cards matching the query
        
        Args:
            query: Anki search query (e.g., "deck:current", "is:due")
            limit: Maximum number of cards to return
            
        Returns:
            List of card objects with id, question, and answer
        """
        try:
            # First, find card IDs matching the query
            response = requests.post(
                self.anki_connect_url,
                json={
                    "action": "findCards",
                    "version": 6,
                    "params": {
                        "query": query
                    }
                }
            )
            result = response.json()
            
            if "error" in result:
                logger.warning(f"Failed to find cards: {result.get('error')}")
                return []
            
            card_ids = result["result"][:limit]
            
            if not card_ids:
                return []
            
            # Get card info for the found IDs
            response = requests.post(
                self.anki_connect_url,
                json={
                    "action": "cardsInfo",
                    "version": 6,
                    "params": {
                        "cards": card_ids
                    }
                }
            )
            result = response.json()
            
            if "error" in result:
                logger.warning(f"Failed to get card info: {result.get('error')}")
                return []
            
            cards_info = result["result"]
            
            # Format the cards
            cards = []
            for card_info in cards_info:
                cards.append({
                    "cardId": card_info["cardId"],
                    "front": card_info["fields"]["Front"]["value"],
                    "back": card_info["fields"]["Back"]["value"],
                    "due": card_info.get("due", ""),
                    "deck": card_info.get("deckName", "")
                })
            
            return cards
        except requests.RequestException as e:
            logger.error(f"Request error getting cards: {str(e)}")
            return []

class MCPAnkiServer:
    """
    MCP server for Anki that communicates over stdio
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the MCP Anki Server
        
        Args:
            verbose: Whether to print detailed information
        """
        self.verbose = verbose
        self.question_generator = QuestionGeneratorAgent(verbose=verbose)
        self.card_generator = AnkiCardGenerator()
        self.anki_connector = AnkiConnector()
        
        # Resource handlers
        self.resource_handlers = {
            "anki://search/deckcurrent": self._handle_deck_current,
            "anki://search/isdue": self._handle_is_due,
            "anki://search/isnew": self._handle_is_new
        }
        
        # Tool handlers
        self.tool_handlers = {
            "update_cards": self._handle_update_cards,
            "add_card": self._handle_add_card,
            "get_due_cards": self._handle_get_due_cards,
            "get_new_cards": self._handle_get_new_cards,
            "generate_cards_from_topic": self._handle_generate_cards
        }
    
    def _handle_deck_current(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle anki://search/deckcurrent resource"""
        cards = self.anki_connector.get_cards("deck:current")
        return {"cards": cards}
    
    def _handle_is_due(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle anki://search/isdue resource"""
        cards = self.anki_connector.get_cards("is:due")
        return {"cards": cards}
    
    def _handle_is_new(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle anki://search/isnew resource"""
        cards = self.anki_connector.get_cards("is:new")
        return {"cards": cards}
    
    def _handle_update_cards(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle update_cards tool"""
        answers = params.get("answers", [])
        results = []
        
        for answer in answers:
            card_id = answer.get("cardId")
            ease = answer.get("ease")
            
            if not card_id or not ease:
                results.append({"success": False, "error": "Missing cardId or ease"})
                continue
            
            success = self.anki_connector.update_card(card_id, ease)
            results.append({"success": success, "cardId": card_id})
        
        return {"results": results}
    
    def _handle_add_card(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle add_card tool"""
        front = params.get("front", "")
        back = params.get("back", "")
        deck = params.get("deck", "Default")
        
        if not front or not back:
            return {"success": False, "error": "Missing front or back content"}
        
        card_id = self.anki_connector.add_card(front, back, deck)
        success = card_id > 0
        
        return {
            "success": success,
            "cardId": card_id if success else 0
        }
    
    def _handle_get_due_cards(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_due_cards tool"""
        num = params.get("num", 10)
        cards = self.anki_connector.get_cards("is:due", num)
        return {"cards": cards}
    
    def _handle_get_new_cards(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_new_cards tool"""
        num = params.get("num", 10)
        cards = self.anki_connector.get_cards("is:new", num)
        return {"cards": cards}
    
    def _handle_generate_cards(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generate_cards_from_topic tool - our custom tool"""
        topic = params.get("topic", "")
        num_questions = params.get("num", 5)
        deck_name = params.get("deck", "Auto-Anki")
        
        if not topic:
            return {"success": False, "error": "Missing topic"}
        
        try:
            # Generate questions and answers
            questions_data, answers_data = self.question_generator.generate(
                topic=topic,
                num_questions=num_questions,
                save=False
            )
            
            # Check for errors
            if "error" in questions_data or "error" in answers_data:
                error = questions_data.get("error", "") or answers_data.get("error", "")
                return {"success": False, "error": error}
            
            # Convert to Anki cards
            cards = self.card_generator.convert_to_anki_cards(questions_data, answers_data)
            
            # Add cards to Anki
            added_cards = []
            for card in cards:
                card_id = self.anki_connector.add_card(card["front"], card["back"], deck_name)
                if card_id > 0:
                    added_cards.append({
                        "cardId": card_id,
                        "front": card["front"],
                        "back": card["back"]
                    })
            
            return {
                "success": True,
                "topic": topic,
                "cards_generated": len(cards),
                "cards_added": len(added_cards),
                "added_cards": added_cards
            }
        except Exception as e:
            logger.error(f"Error generating cards: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an MCP request
        
        Args:
            request: MCP request object
            
        Returns:
            Response object
        """
        request_type = request.get("type")
        
        if request_type == "resource":
            # Handle resource request
            resource_name = request.get("name", "")
            params = request.get("params", {})
            
            if resource_name in self.resource_handlers:
                try:
                    return {
                        "type": "resource_response",
                        "status": "success",
                        "body": self.resource_handlers[resource_name](params)
                    }
                except Exception as e:
                    logger.error(f"Error handling resource {resource_name}: {str(e)}")
                    return {
                        "type": "resource_response",
                        "status": "error",
                        "error": str(e)
                    }
            else:
                return {
                    "type": "resource_response",
                    "status": "error",
                    "error": f"Unknown resource: {resource_name}"
                }
        
        elif request_type == "tool":
            # Handle tool request
            tool_name = request.get("name", "")
            params = request.get("params", {})
            
            if tool_name in self.tool_handlers:
                try:
                    result = self.tool_handlers[tool_name](params)
                    return {
                        "type": "tool_response",
                        "status": "success",
                        "body": result
                    }
                except Exception as e:
                    logger.error(f"Error handling tool {tool_name}: {str(e)}")
                    return {
                        "type": "tool_response",
                        "status": "error",
                        "error": str(e)
                    }
            else:
                return {
                    "type": "tool_response",
                    "status": "error",
                    "error": f"Unknown tool: {tool_name}"
                }
        
        elif request_type == "manifest":
            # Return manifest with supported resources and tools
            return {
                "type": "manifest_response",
                "body": {
                    "schema_version": "1.0",
                    "name": "auto-anki",
                    "display_name": "Auto-Anki",
                    "description": "An MCP server that generates Anki cards from topics and interacts with Anki",
                    "resources": [
                        {
                            "name": "anki://search/deckcurrent",
                            "description": "Returns all cards from current deck"
                        },
                        {
                            "name": "anki://search/isdue",
                            "description": "Returns cards in review and learning waiting to be studied"
                        },
                        {
                            "name": "anki://search/isnew",
                            "description": "Returns all unseen cards"
                        }
                    ],
                    "tools": [
                        {
                            "name": "update_cards",
                            "description": "Marks cards with given card IDs as answered and gives them an ease score",
                            "params": {
                                "answers": {
                                    "type": "array",
                                    "description": "Array of objects with cardId and ease fields",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "cardId": {
                                                "type": "number",
                                                "description": "The ID of the card"
                                            },
                                            "ease": {
                                                "type": "number",
                                                "description": "Ease score between 1 (Again) and 4 (Easy)"
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        {
                            "name": "add_card",
                            "description": "Creates a new card in the specified Anki deck",
                            "params": {
                                "front": {
                                    "type": "string",
                                    "description": "Front of card"
                                },
                                "back": {
                                    "type": "string",
                                    "description": "Back of card"
                                },
                                "deck": {
                                    "type": "string",
                                    "description": "Name of the deck",
                                    "default": "Default"
                                }
                            }
                        },
                        {
                            "name": "get_due_cards",
                            "description": "Returns a number of cards currently due for review",
                            "params": {
                                "num": {
                                    "type": "number",
                                    "description": "Number of cards",
                                    "default": 10
                                }
                            }
                        },
                        {
                            "name": "get_new_cards",
                            "description": "Returns a number of new cards",
                            "params": {
                                "num": {
                                    "type": "number",
                                    "description": "Number of cards",
                                    "default": 10
                                }
                            }
                        },
                        {
                            "name": "generate_cards_from_topic",
                            "description": "Generates Anki cards from a topic using AI",
                            "params": {
                                "topic": {
                                    "type": "string",
                                    "description": "The topic to generate questions and answers about"
                                },
                                "num": {
                                    "type": "number",
                                    "description": "Number of questions to generate",
                                    "default": 5
                                },
                                "deck": {
                                    "type": "string",
                                    "description": "Name of the deck to add cards to",
                                    "default": "Auto-Anki"
                                }
                            }
                        }
                    ]
                }
            }
        
        else:
            return {
                "type": "error",
                "error": f"Unknown request type: {request_type}"
            }
    
    def start(self):
        """
        Start the MCP server
        """
        logger.info("Starting Auto-Anki MCP Server")
        
        # Print the manifest to stderr for debugging
        if self.verbose:
            manifest = self.handle_request({"type": "manifest"})
            logger.info(f"Manifest: {json.dumps(manifest, indent=2)}")
        
        try:
            # Check Anki connection
            if not self.anki_connector.check_connection():
                logger.error("Cannot connect to Anki. Make sure Anki is running with AnkiConnect plugin installed.")
                sys.stderr.write("Cannot connect to Anki. Make sure Anki is running with AnkiConnect plugin installed.\n")
                return
            
            logger.info("Connected to AnkiConnect successfully")
            
            # Main MCP communication loop
            while True:
                # Read a line from stdin
                line = sys.stdin.readline()
                
                if not line:
                    break
                
                # Parse the request
                try:
                    request = json.loads(line)
                    logger.debug(f"Received request: {json.dumps(request)}")
                    
                    # Handle the request
                    response = self.handle_request(request)
                    
                    # Write the response to stdout
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON: {line}")
                    sys.stdout.write(json.dumps({
                        "type": "error",
                        "error": "Invalid JSON request"
                    }) + "\n")
                    sys.stdout.flush()
                except Exception as e:
                    logger.error(f"Error handling request: {str(e)}")
                    sys.stdout.write(json.dumps({
                        "type": "error",
                        "error": str(e)
                    }) + "\n")
                    sys.stdout.flush()
        except KeyboardInterrupt:
            logger.info("MCP server shutting down")
        except Exception as e:
            logger.error(f"Unhandled exception: {str(e)}")
            sys.stderr.write(f"Error: {str(e)}\n")
            sys.stderr.flush()

if __name__ == "__main__":
    # Just run the MCP server directly
    server = MCPAnkiServer(verbose=True)
    server.start() 