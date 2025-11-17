"""
Interactive Chat with Memory Management for Mental Health Pipeline

This system maintains conversation context through:
1. User discussion summaries (in their words)
2. Doctor's clinical notes and insights
3. Context-aware routing to the main pipeline
4. Memory usage monitoring
"""

import sys
import time
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import logging
warnings.filterwarnings('ignore')
# Suppress transformers messages about padding
logging.getLogger("transformers").setLevel(logging.ERROR)

# Import the main pipeline
sys.path.insert(0, str(Path(__file__).parent))
from main_pipeline import load_model, generate


class ConversationMemory:
    """Manages conversation history and context."""

    def __init__(self, max_context_tokens: int = 2048):
        self.user_summaries: List[str] = []
        self.doctor_notes: List[str] = []
        self.conversation_turns: int = 0
        self.max_context_tokens = max_context_tokens
        self.current_session_start = datetime.now()

    def add_turn(self, user_summary: str, doctor_note: str):
        """Add a conversation turn to memory."""
        self.user_summaries.append(user_summary)
        self.doctor_notes.append(doctor_note)
        self.conversation_turns += 1

    def get_formatted_context(self) -> str:
        """Get the formatted context for the next turn."""
        if not self.user_summaries:
            return ""

        # Combine recent summaries (keep last 3-5 turns for relevance)
        recent_user = " ".join(self.user_summaries[-5:])
        recent_notes = " ".join(self.doctor_notes[-3:])

        context = f"""Previous context:

Summary of user discussion: {recent_user}

Doctor context: {recent_notes}"""

        return context

    def estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count (4 chars per token average)."""
        return len(text) // 4

    def get_memory_usage(self) -> Dict:
        """Get current memory usage statistics."""
        context = self.get_formatted_context()
        estimated_tokens = self.estimate_tokens(context)

        return {
            "turns": self.conversation_turns,
            "context_tokens": estimated_tokens,
            "max_tokens": self.max_context_tokens,
            "usage_percentage": (estimated_tokens / self.max_context_tokens) * 100,
            "remaining_tokens": self.max_context_tokens - estimated_tokens
        }


class InteractiveMentalHealthChat:
    """Interactive chat system with memory and context management."""

    def __init__(self):
        print("="*60)
        print("INTERACTIVE MENTAL HEALTH CHAT SYSTEM")
        print("="*60)
        print("\nInitializing system components...")

        # Load the model once
        self.model, self.tokenizer = load_model()

        # Initialize memory
        self.memory = ConversationMemory()

        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("System ready for conversation")
        print("\n" + "="*60)

    def summarize_user_input(self, user_input: str) -> str:
        """Generate a summary of user input in their words."""
        prompt = f"""Summarize the following patient statement in their own words, keeping key concerns and emotions:

Patient said: "{user_input}"

Brief summary (keep the patient's perspective and language):"""

        summary = generate(self.model, self.tokenizer, prompt)

        # Keep it concise
        if len(summary) > 200:
            summary = summary[:197] + "..."

        return summary.strip()

    def extract_doctor_notes(self, professional_opinion: str, user_input: str) -> str:
        """Extract key clinical notes from the professional opinion."""
        prompt = f"""Based on this assessment, write a brief clinical note (2-3 sentences max):

Assessment: {professional_opinion}

Clinical note (key observations and concerns only):"""

        note = generate(self.model, self.tokenizer, prompt)

        # Keep notes concise
        if len(note) > 150:
            note = note[:147] + "..."

        return note.strip()

    def run_contextual_pipeline(self, user_input: str, context: str) -> Dict:
        """Run the pipeline with context included."""

        # Combine context with new input
        if context:
            full_input = f"{user_input}\n\n{context}"
        else:
            full_input = user_input


        # Component 1: Clinical description
        clinical_prompt = f"""Convert this to third-person clinical language:
"{full_input}"

Change "I" to "The patient", keep it concise and clinical."""

        clinical_description = generate(self.model, self.tokenizer, clinical_prompt)

        # Component 2: Professional opinion (includes context awareness)
        opinion_prompt = f"""As a mental health professional, provide a brief assessment of:
{clinical_description}

Identify key concerns and provide professional opinion."""

        professional_opinion = generate(self.model, self.tokenizer, opinion_prompt)

        # Component 3: Final response
        response_prompt = f"""You are an empathetic counselor in an ongoing conversation. Using the professional context below, respond helpfully to the patient's concern.

Original concern: {user_input}

Professional context: {professional_opinion}

Provide a compassionate, helpful response that acknowledges any previous discussions if relevant:"""

        final_response = generate(self.model, self.tokenizer, response_prompt)

        return {
            "clinical_description": clinical_description,
            "professional_opinion": professional_opinion,
            "final_response": final_response
        }

    def display_memory_status(self):
        """Display current memory usage."""
        stats = self.memory.get_memory_usage()

        # Create a simple text-based progress bar
        usage_pct = min(stats['usage_percentage'], 100)
        bar_length = 30
        filled = int(bar_length * usage_pct / 100)
        bar = '█' * filled + '░' * (bar_length - filled)

        print(f"\nMemory Status:")
        print(f"   Turns: {stats['turns']}")
        print(f"   Context: [{bar}] {usage_pct:.1f}%")
        print(f"   Tokens: {stats['context_tokens']}/{stats['max_tokens']}")

        if usage_pct > 80:
            print("   Warning: Approaching context limit")
        elif usage_pct > 90:
            print("   Critical: Context nearly full, older details may be lost")

    def handle_special_commands(self, user_input: str) -> bool:
        """Handle special commands. Returns True if handled."""
        command = user_input.lower().strip()

        if command == '/memory':
            self.display_memory_status()
            print("\nContext being maintained:")
            print("-"*40)
            if self.memory.user_summaries:
                print("Recent user concerns:")
                for i, summary in enumerate(self.memory.user_summaries[-3:], 1):
                    print(f"  {i}. {summary}")
                print("\nRecent clinical notes:")
                for i, note in enumerate(self.memory.doctor_notes[-3:], 1):
                    print(f"  {i}. {note}")
            else:
                print("  No conversation history yet")
            return True

        elif command == '/clear':
            self.memory = ConversationMemory()
            print("\nMemory cleared. Starting fresh conversation.")
            return True

        elif command == '/help':
            print("\nAvailable Commands:")
            print("  /memory - View current memory and context")
            print("  /clear  - Clear conversation memory")
            print("  /help   - Show this help message")
            print("  /quit   - Exit the chat")
            return True

        return False

    def chat_loop(self):
        """Main interactive chat loop."""
        print("\nMental Health Support Chat")
        print("="*60)
        print("Type '/help' for commands, '/quit' to exit")
        print("="*60)

        print("\nHello! I'm here to provide mental health support.")
        print("Please share what's on your mind.\n")

        while True:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Check for exit
            if user_input.lower() in ['/quit', 'exit', 'quit']:
                print("\nThank you for our conversation. Take care!")
                break

            # Handle special commands
            if user_input.startswith('/'):
                if self.handle_special_commands(user_input):
                    continue

            # Skip empty input
            if not user_input:
                print("Please share your thoughts or concerns.")
                continue

            # Process the input
            print("\nProcessing...")

            # Get context from memory
            context = self.memory.get_formatted_context()

            # Run the pipeline with context
            result = self.run_contextual_pipeline(user_input, context)

            # Display response
            print("\n" + "="*60)
            print("Counselor:")
            print("-"*60)
            print(result['final_response'])
            print("="*60)

            # Update memory with summaries
            user_summary = self.summarize_user_input(user_input)
            doctor_note = self.extract_doctor_notes(
                result['professional_opinion'],
                user_input
            )

            self.memory.add_turn(user_summary, doctor_note)

            # Show memory status periodically
            if self.memory.conversation_turns % 3 == 0:
                self.display_memory_status()

            # Check if approaching context limit
            stats = self.memory.get_memory_usage()
            if stats['usage_percentage'] > 85:
                print("\nNote: Conversation context is getting long.")
                print("Consider focusing on current concerns or use '/clear' to start fresh.")

    def run(self):
        """Main entry point for the interactive chat."""
        try:
            self.chat_loop()
        except KeyboardInterrupt:
            print("\n\nChat interrupted. Thank you for our conversation!")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please restart the chat system.")
        finally:
            # Cleanup
            print("\n" + "="*60)
            print(f"Session ended: {self.session_id}")
            print(f"Total turns: {self.memory.conversation_turns}")
            print("="*60)


def main():
    """Main function to start the interactive chat."""
    print("\n" + "="*80)
    print("MENTAL HEALTH INTERACTIVE CHAT SYSTEM")
    print("="*80)
    print("\nThis system is for supportive conversation only.")
    print("For crisis situations, please contact emergency services or a crisis hotline.")
    print("\n" + "="*80)

    # Create and run the chat system
    chat = InteractiveMentalHealthChat()
    chat.run()


if __name__ == "__main__":
    main()