import os
import sys
import json
from src.question_generator import QuestionGeneratorAgent
from src.auto_anki import AnkiCardGenerator, AnkiConnector

def read_text_file(file_path):
    """Read content from a text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demo_file = os.path.join(script_dir, 'demo', 'q_demo', 'q_demo.txt')
    
    # Check if file exists
    if not os.path.exists(demo_file):
        print(f"Error: Demo file not found at {demo_file}")
        return
    
    # Read the content from q_demo.txt
    content = read_text_file(demo_file)
    print(f"Loaded content from {demo_file}")
    print("-" * 50)
    print(content)
    print("-" * 50)
    
    # Use the content directly as input for the question generator
    print("Generating questions based on the content from q_demo.txt")
    
    try:
        # Initialize QuestionGeneratorAgent
        question_generator = QuestionGeneratorAgent(verbose=True)
        
        # Generate questions and answers using the content from the file
        questions_data, answers_data = question_generator.generate(
            topic=content,  # Using the content directly as the topic
            num_questions=5,
            save=True
        )
        
        # Check for errors in questions_data
        if "error" in questions_data:
            print(f"Error generating questions: {questions_data['error']}")
            return
            
        print(f"Generated {len(questions_data.get('questions', []))} questions")
        
        # Check for errors in answers_data
        if "error" in answers_data:
            print(f"Error generating answers: {answers_data['error']}")
            # If we have questions but failed to get answers, we can still try to proceed
            if not questions_data.get("questions"):
                return
            print("Will attempt to continue with questions only...")
        
        # Initialize AnkiCardGenerator
        card_generator = AnkiCardGenerator()
        
        try:
            # Convert to Anki cards
            cards = card_generator.convert_to_anki_cards(questions_data, answers_data)
            print(f"Converted to {len(cards)} Anki cards")
            
            # Check if Anki is running with AnkiConnect
            anki_connector = AnkiConnector()
            if anki_connector.check_connection():
                print("Connected to Anki. Adding cards...")
                
                # Create or get deck
                deck_name = "Demo-Content-Deck"
                anki_connector.create_deck(deck_name)
                
                # Add cards to Anki
                added_cards = []
                for card in cards:
                    card_id = anki_connector.add_card(card["front"], card["back"], deck_name)
                    if card_id > 0:
                        added_cards.append(card_id)
                
                print(f"Added {len(added_cards)} cards to Anki deck '{deck_name}'")
            else:
                print("Cannot connect to Anki. Make sure Anki is running with AnkiConnect plugin installed.")
                print("Printing cards instead:")
                for i, card in enumerate(cards):
                    print(f"\nCard {i+1}:")
                    print("Front:")
                    print(card["front"])
                    print("\nBack:")
                    print(card["back"])
                    print("-" * 50)
        except Exception as e:
            print(f"Error converting to Anki cards: {str(e)}")
            return
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure you have installed AutoGen. Run: pip install ag2[openai]")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 