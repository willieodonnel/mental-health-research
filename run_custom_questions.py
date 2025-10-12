"""
Custom question runner - easily test your own mental health questions
"""

from mental_health_inference import MentalHealthInferencePipeline

# Add your own questions here!
CUSTOM_QUESTIONS = [
    # Example questions - replace with your own
    "How can I cope with feeling overwhelmed at work?",

    "I've been having trouble connecting with my friends lately. What should I do?",

    "What are some strategies for managing stress during difficult times?",

    # Add more questions below:
    # "Your question here?",
]


def main():
    """Run pipeline on custom questions."""

    print("\n" + "="*80)
    print("🧠 Mental Health Custom Question Runner")
    print("="*80)

    if not CUSTOM_QUESTIONS:
        print("\n⚠️  No questions found!")
        print("Edit 'run_custom_questions.py' and add your questions to the CUSTOM_QUESTIONS list.")
        return

    print(f"\n📝 Found {len(CUSTOM_QUESTIONS)} question(s) to process")

    # Initialize pipeline
    print("\n" + "-"*80)
    pipeline = MentalHealthInferencePipeline()
    print("-"*80)

    # Process questions
    results = pipeline.process_batch(
        CUSTOM_QUESTIONS,
        output_file="custom_responses.jsonl",
        max_new_tokens=512,
        temperature=0.7,  # Adjust for creativity (0.1-1.0)
        top_p=0.9,
        top_k=50,
    )

    # Print detailed results
    print("\n" + "="*80)
    print("📊 DETAILED RESULTS")
    print("="*80)

    for i, result in enumerate(results, 1):
        print(f"\n{'─'*80}")
        print(f"Question {i}/{len(results)}")
        print(f"{'─'*80}")
        print(f"\n❓ {result['question']}\n")
        print(f"💬 {result['response']}\n")
        print(f"⏱️  Generated in {result['metadata']['generation_time_seconds']:.2f}s")
        print(f"🚀 Speed: {result['metadata']['tokens_per_second']:.1f} tokens/sec")
        print(f"📝 Tokens: {result['metadata']['tokens_generated']}")

    print("\n" + "="*80)
    print("✅ All responses saved to: custom_responses.jsonl")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
