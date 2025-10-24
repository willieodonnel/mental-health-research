"""
Quick script to view evaluation results in a readable format
"""

import json

# Read the JSONL file
with open('results_mistral_local.jsonl', 'r', encoding='utf-8') as f:
    results = [json.loads(line) for line in f if line.strip()]

print(f"\n{'='*80}")
print(f"EVALUATION RESULTS: {len(results)} Questions")
print(f"{'='*80}\n")

for i, result in enumerate(results, 1):
    print(f"\n{'='*80}")
    print(f"Question {i}/{len(results)}")
    print(f"{'='*80}")

    print(f"\nQuestion: {result['question'][:200]}...")
    print(f"\nMistral's Response: {result['local_mistral_response'][:300]}...")

    print(f"\nScores:")
    if result['scores']:
        for metric, score in result['scores'].items():
            print(f"  {metric}: {score}/10")
        avg = sum(result['scores'].values()) / len(result['scores'])
        print(f"  Average: {avg:.2f}/10")

    print(f"\nJudge Explanation: {result.get('explanation', 'N/A')[:200]}...")

    input("\nPress Enter for next question...")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}\n")

# Calculate overall stats
all_scores = {metric: [] for metric in ['Active Listening', 'Empathy & Validation',
                                         'Safety & Trustworthiness', 'Open-mindedness & Non-judgment',
                                         'Clarity & Encouragement', 'Boundaries & Ethical',
                                         'Holistic Approach']}

for result in results:
    if result['scores']:
        for metric, score in result['scores'].items():
            if score > 0:  # Exclude failed evaluations
                all_scores[metric].append(score)

print("Average Scores:")
for metric, scores in all_scores.items():
    if scores:
        avg = sum(scores) / len(scores)
        print(f"  {metric}: {avg:.2f}/10")
