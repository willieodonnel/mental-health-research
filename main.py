"""
LEGACY Main Script - GPT-4 Three-Stage Pipeline

WARNING: This script uses the LEGACY three-stage GPT-4 pipeline, NOT the main inference method.

For the recommended Mistral-7B local inference, use:
- mental_health_inference.py (batch mode)
- run_custom_questions.py (custom questions)
- Interactive mode via MentalHealthInferencePipeline().interactive_mode()

This script demonstrates the three-stage GPT-4 pipeline (requires OpenAI API key and incurs costs).
For evaluation, use evaluation_local.py with Mistral-7B inference.
"""

from pipeline import mental_health_pipeline

if __name__ == "__main__":
    # Example user input
    example_input = """I've been feeling really anxious lately. I can't seem to focus on anything
    and I keep forgetting things. Yesterday I forgot my meeting and today I left my keys in the door.
    I'm worried something is wrong with my memory."""

    print("=" * 60)
    print("MENTAL HEALTH SUPPORT PIPELINE - DEMO")
    print("=" * 60)
    print()
    print("Running pipeline with example input...")
    print()

    # Run the pipeline
    response = mental_health_pipeline(example_input)

    print()
    print("=" * 60)
    print("For evaluation, use:")
    print("  Unofficial (quick test): python evaluation.py --mode unofficial --num_samples 20")
    print("  Official (200 test set): python evaluation.py --mode official --questions_jsonl test_set_200.jsonl")
    print("=" * 60)
