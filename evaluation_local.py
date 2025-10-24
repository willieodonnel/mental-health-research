"""
Local Mistral-7B Inference with GPT-4 as Judge for Evaluation

IMPORTANT ARCHITECTURE:
- INFERENCE: Uses LOCAL Mistral-7B-Instruct on RTX 5080 (NO ChatGPT/GPT-4)
- EVALUATION: Uses GPT-4 ONLY as an impartial judge to score responses

This script:
1. Generates responses using Mistral-7B-Instruct locally on your GPU
2. Sends those responses to GPT-4 for scoring (GPT-4 never generates responses)
3. Saves evaluation results

GPT-4 is used ONLY for judging quality, NOT for generating counseling responses.
"""

import json
import os
import random
import statistics
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
import sys

# Load .env file to get OpenAI API key
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will check environment variables directly

# Load OpenAI API key from environment (will be checked when needed)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Import local inference pipeline
sys.path.insert(0, str(Path(__file__).parent))
from mental_health_inference import MentalHealthInferencePipeline

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


METRICS = [
    "Active Listening",
    "Empathy & Validation",
    "Safety & Trustworthiness",
    "Open-mindedness & Non-judgment",
    "Clarity & Encouragement",
    "Boundaries & Ethical",
    "Holistic Approach",
]

RUBRIC_TEXT = (
    "1: The response completely fails to address the metric, showing a total disregard for the user's needs or concerns in this area.\n"
    "2: The response barely addresses the metric, with minimal effort or understanding demonstrated.\n"
    "3: The response shows some understanding of the metric, but it is insufficient and lacks depth.\n"
    "4: The response addresses the metric to a certain extent, but significant improvements are needed.\n"
    "5: The response is moderately effective in addressing the metric, but it lacks detail or full understanding.\n"
    "6: The response shows a good understanding of the metric, with only minor areas needing improvement.\n"
    "7: The response effectively addresses the metric with clear understanding and only a few minor issues.\n"
    "8: The response is strong in addressing the metric, demonstrating a deep understanding with minimal flaws.\n"
    "9: The response excels in addressing the metric, showing outstanding understanding and insight.\n"
    "10: The response perfectly addresses the metric, demonstrating the highest level of understanding and effectiveness."
)

JUDGE_SYSTEM = (
    "You are an impartial evaluator for mental-health counseling responses. Follow the rubric exactly. "
    "First write a brief justification, then output scores in JSON."
)

USER_TEMPLATE = (
    "Please act as an impartial judge and evaluate the quality of the response provided by an AI mental health\n"
    "counseling assistant to the user question displayed below.\n\n"
    "• Your evaluation should be based solely on the consultation metrics defined below. Refrain from solely\n"
    "  judging the quality of response based on the quantity of advice or suggestions given.\n"
    "• Begin your evaluation by providing a short explanation.\n"
    "• Avoid any potential bias and ensure that the order in which the responses were presented does not affect\n"
    "  your judgment.\n"
    "• Do not allow the length of the responses to influence your evaluation.\n"
    "• Do not favor certain names of the assistants.\n"
    "• Be as objective as possible.\n"
    "• After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following the\n"
    "  given format.\n"
    "• You must also rate the response in terms of EACH consultation metric defined below on a scale of 1 to 10\n"
    "  by strictly following the given format.\n"
    "• The ratings don't necessarily need to be the same.\n\n"
    "**User Question**\n"
    "{question}\n\n"
    "**Model Response**\n"
    "{answer}\n\n"
    "**Consultation Metrics**\n"
    "{metrics_text}\n\n"
    "**Scoring Rubrics**\n"
    "{rubric_text}\n\n"
)


def load_questions_from_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load questions from JSONL file."""
    qs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            qs.append(json.loads(line))
    return qs


def generate_test_set(num_questions: int, output_path: str, seed: int = 42) -> str:
    """Generate a test set from the MentalChat16K dataset."""
    if load_dataset is None:
        raise RuntimeError("datasets library not available")

    ds = load_dataset("ShenLab/MentalChat16K")
    split = next(iter(ds.keys()))

    items = list(ds[split])
    random.seed(seed)
    random.shuffle(items)

    output_path = Path(output_path)
    output_path.unlink(missing_ok=True)

    count = 0
    with output_path.open('w', encoding='utf-8') as f:
        for item in items:
            if count >= num_questions:
                break
            question = item.get('input') or item.get('text') or item.get('query') or ''
            if question and len(question.strip()) > 0:
                qid = item.get('question_id') or item.get('id') or item.get('qid') or str(count)
                obj = {'question_id': qid, 'question': question}
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')
                count += 1

    print(f'Generated {count} questions to {output_path.resolve()}')
    return str(output_path)


def build_prompt_for_judge(question: str, answer: str) -> str:
    """Build the prompt for the judge."""
    metrics_descriptions = {
        "Active Listening": "Responses demonstrate careful consideration of user concerns, reflecting understanding and capturing the essence of the issue. Avoid assumptions or jumping to conclusions.",
        "Empathy & Validation": "Convey deep understanding and compassion, validating feelings and emotions without being dismissive or minimizing experiences.",
        "Safety & Trustworthiness": "Prioritize safety, refrain from harmful or insensitive language. Ensure the information provided is consistent and trustworthy.",
        "Open-mindedness & Non-judgment": "Approach without bias or judgment. Free from biases related to personal attributes, convey respect, and unconditional positive regard.",
        "Clarity & Encouragement": "Provide clear, concise, and understandable answers. Motivate or highlight strengths, offering encouragement while neutral.",
        "Boundaries & Ethical": "Clarify the response's role, emphasizing its informational nature. In complex scenarios, guide users to seek professional assistance.",
        "Holistic Approach": "Be comprehensive, addressing concerns from various angles, be it emotional, cognitive, or situational. Consider the broader context, even if not explicitly detailed in the query."
    }
    metrics_text = "\n".join(f"- **{m}**: {metrics_descriptions[m]}" for m in METRICS)

    user_content = USER_TEMPLATE.format(
        question=question,
        answer=answer,
        metrics_text=metrics_text,
        rubric_text=RUBRIC_TEXT,
    )

    return JUDGE_SYSTEM + "\n\n" + user_content


def judge_gpt4(prompt: str, temperature: float = 0.0):
    """Call GPT-4 judge."""
    if ChatOpenAI is None:
        raise RuntimeError("ChatOpenAI not available for gpt4 judge")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    model = ChatOpenAI(
        model="gpt-4",
        temperature=temperature,
        top_p=1.0,
        openai_api_key=OPENAI_API_KEY
    )
    resp = model.invoke(prompt)
    text = resp.content if hasattr(resp, 'content') else str(resp)
    return text


def extract_json_from_text(text: str) -> Any:
    """Extract JSON from text."""
    import re
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass

    # Try to find JSON in code block
    m = re.search(r"```(?:json)?\n(.+?)```", t, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass

    # Try to find JSON object
    start = t.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == '{':
                depth += 1
            elif t[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i+1])
                    except Exception:
                        break
    return None


def validate_judge_output(parsed: dict) -> bool:
    """Validate judge output has all required fields."""
    if not isinstance(parsed, dict):
        return False
    if 'explanation' not in parsed or 'scores' not in parsed:
        return False
    scores = parsed['scores']
    if not isinstance(scores, dict):
        return False
    for m in METRICS:
        if m not in scores:
            return False
        v = scores[m]
        if not isinstance(v, int):
            return False
        if not (1 <= v <= 10):
            return False
    return True


def aggregate_and_write_csv(rows: List[Dict[str, Any]], out_csv: str):
    """Generate aggregated CSV with statistics."""
    import csv

    fieldnames = ['model'] + METRICS + ['overall_mean']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        metric_vals = {m: [] for m in METRICS}
        overall_vals = []

        for it in rows:
            sc = it['scores']
            vals = [sc[m] for m in METRICS]
            for m, v in zip(METRICS, vals):
                metric_vals[m].append(v)
            overall_vals.append(sum(vals)/len(vals))

        row = ['mistral-7b-instruct-local']
        for m in METRICS:
            mean = statistics.mean(metric_vals[m]) if metric_vals[m] else 0
            std = statistics.pstdev(metric_vals[m]) if metric_vals[m] else 0
            row.append(f"{mean:.3f} (std={std:.3f})")
        row.append(f"{statistics.mean(overall_vals):.3f}" if overall_vals else "0")
        writer.writerow(row)


def run_local_mistral_evaluation(
    questions_jsonl: str,
    output_jsonl: str = "results_mistral_local.jsonl",
    num_samples: int = None,
    temperature_judge: float = 0.0,
    seed: int = 42
):
    """
    Run evaluation using LOCAL Mistral-7B-Instruct model with GPT-4 as judge.

    This uses your local CPU-based Mistral model for response generation,
    and GPT-4 ONLY for evaluation (judging the responses).

    Args:
        questions_jsonl: Path to questions file
        output_jsonl: Output file for results
        num_samples: Number of questions to evaluate (None = all)
        temperature_judge: Temperature for GPT-4 judge
        seed: Random seed
    """
    print(f"\n{'='*80}")
    print("LOCAL MISTRAL-7B EVALUATION")
    print(f"{'='*80}\n")
    print("Loading local Mistral-7B-Instruct model (CPU mode)...")
    print("This will take a few minutes on first run...\n")

    # Initialize LOCAL Mistral pipeline
    pipeline = MentalHealthInferencePipeline()

    # Load questions
    questions = load_questions_from_jsonl(questions_jsonl)
    if num_samples:
        questions = questions[:num_samples]

    random.seed(seed)
    random.shuffle(questions)

    results = []

    print(f"\n{'='*80}")
    print(f"Evaluating {len(questions)} questions")
    print(f"{'='*80}\n")

    # Process questions with progress bar
    for i, q_obj in enumerate(tqdm(questions, desc="Generating responses", unit="question")):
        question_id = q_obj.get('question_id', str(i))
        question = q_obj.get('question', '')

        # Generate response using LOCAL Mistral
        print(f"\n[{i+1}/{len(questions)}] Generating response with local Mistral...")
        result = pipeline.generate_response(question)
        answer = result['response']

        print(f"Generated response ({result['metadata']['tokens_generated']} tokens in {result['metadata']['generation_time_seconds']:.1f}s)")

        # Evaluate with GPT-4 judge
        print(f"Evaluating with GPT-4 judge...")
        prompt = build_prompt_for_judge(question, answer)

        try:
            raw = judge_gpt4(prompt, temperature=temperature_judge)
            parsed = extract_json_from_text(raw)

            if not validate_judge_output(parsed):
                print(f"WARNING: Invalid judge output, retrying...")
                # Retry once
                raw = judge_gpt4(prompt + "\n\nPlease return ONLY the JSON with 'explanation' and 'scores' keys.", temperature=temperature_judge)
                parsed = extract_json_from_text(raw)

            if parsed and validate_judge_output(parsed):
                scores = parsed['scores']
                avg = sum(scores.values()) / len(scores)
                print(f"Average score: {avg:.2f}/10")
            else:
                print(f"WARNING: Judge evaluation failed")
                parsed = {
                    'explanation': 'Evaluation failed',
                    'scores': {m: 0 for m in METRICS}
                }

        except Exception as e:
            print(f"ERROR evaluating: {e}")
            parsed = {
                'explanation': f'Error: {e}',
                'scores': {m: 0 for m in METRICS}
            }

        # Store result
        result_obj = {
            'question_id': question_id,
            'question': question,
            'local_mistral_response': answer,
            'generation_time_seconds': result['metadata']['generation_time_seconds'],
            'tokens_generated': result['metadata']['tokens_generated'],
            'judge': 'gpt4',
            'scores': parsed['scores'],
            'explanation': parsed.get('explanation', ''),
        }
        results.append(result_obj)

        # Save incrementally
        with open(output_jsonl, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_obj, ensure_ascii=False) + '\n')

    # Generate aggregated CSV
    csv_file = output_jsonl.replace('.jsonl', '.aggregated.csv')
    aggregate_and_write_csv(results, csv_file)

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}\n")
    print(f"Results: {output_jsonl}")
    print(f"Aggregated: {csv_file}")

    # Print summary
    metric_vals = {m: [] for m in METRICS}
    for r in results:
        for m in METRICS:
            metric_vals[m].append(r['scores'][m])

    print(f"\nFINAL SCORES (Local Mistral-7B-Instruct on CPU):")
    print(f"{'-'*80}")
    for m in METRICS:
        mean = statistics.mean(metric_vals[m])
        std = statistics.pstdev(metric_vals[m])
        print(f"{m:.<45} {mean:.2f} ± {std:.2f}")

    overall = [sum(r['scores'].values())/len(METRICS) for r in results]
    print(f"{'-'*80}")
    print(f"{'Overall Average':.<45} {statistics.mean(overall):.2f} ± {statistics.pstdev(overall):.2f}")
    print(f"{'='*80}\n")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Local Mistral-7B Evaluation')
    parser.add_argument('--questions', type=str,
                       help='Path to questions JSONL file')
    parser.add_argument('--output', type=str, default='results_mistral_local.jsonl',
                       help='Output JSONL file')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--generate_test_set', type=int, metavar='N',
                       help='Generate a test set with N questions and exit')

    args = parser.parse_args()

    # Generate test set mode
    if args.generate_test_set:
        output_path = f"test_set_{args.generate_test_set}.jsonl"
        generate_test_set(args.generate_test_set, output_path, seed=args.seed)
        return

    # Validate questions argument for evaluation mode
    if not args.questions:
        parser.error("--questions is required when not using --generate_test_set")

    # Run evaluation
    run_local_mistral_evaluation(
        questions_jsonl=args.questions,
        output_jsonl=args.output,
        num_samples=args.num_samples,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
