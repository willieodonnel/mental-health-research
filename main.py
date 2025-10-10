"""
Main script to run the mental health pipeline.

This script demonstrates the three-stage pipeline for mental health support.
For evaluation, use evaluation.py with --mode unofficial or --mode official.
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
