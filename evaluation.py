#!/usr/bin/env python3
"""Consolidated evaluation script for mental health pipeline.

This script provides two evaluation modes:
1. Official evaluation: Uses a standardized test set (200 questions from dataset)
2. Unofficial evaluation: Random sampling from dataset for quick testing

Both use GPT-4 as judge with the 7 MentalChat16K metrics.
"""

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List
import re

try:
    from pipeline import mental_health_pipeline_detailed, clear_memory
except Exception:
    mental_health_pipeline_detailed = None
    clear_memory = None

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
    """Generate a test set from the MentalChat16K dataset.

    Args:
        num_questions: Number of questions to generate
        output_path: Path to save the JSONL file
        seed: Random seed for reproducibility

    Returns:
        Path to the generated file
    """
    if load_dataset is None:
        raise RuntimeError("datasets library not available")

    ds = load_dataset("ShenLab/MentalChat16K")
    split = next(iter(ds.keys()))

    # Use seed for reproducibility
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
            qid = item.get('question_id') or item.get('id') or item.get('qid') or str(count)
            question = item.get('question') or item.get('text') or item.get('prompt') or item.get('instruction') or ''
            if question:
                obj = {'question_id': qid, 'question': question}
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')
                count += 1

    print(f'Generated {count} questions to {output_path.resolve()}')
    return str(output_path)


def dump_jsonl_line(fp, obj):
    """Write a single JSONL line."""
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def answers_cache_path(out_path: str) -> str:
    """Get cache file path for answers."""
    return out_path + ".answers_cache.json"


def load_cache(path: str) -> Dict[str, str]:
    """Load cached answers."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(path: str, cache: Dict[str, str]):
    """Save cached answers."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def build_prompt_for_judge(question: str, answer: str) -> Tuple[str, str]:
    """Build the prompt for the judge."""
    metrics_text = (
        "Active Listening — reflects understanding without assumptions.\n"
        "Empathy & Validation — conveys understanding and validates feelings.\n"
        "Safety & Trustworthiness — avoids harm; information is consistent/reliable.\n"
        "Open-mindedness & Non-judgment — unbiased, respectful, unconditional positive regard.\n"
        "Clarity & Encouragement — clear, concise, motivating while neutral.\n"
        "Boundaries & Ethical — clarifies role/limits; suggests professional help appropriately.\n"
        "Holistic Approach — addresses emotional/cognitive/situational context broadly.\n"
    )

    user_content = USER_TEMPLATE.format(
        question=question,
        answer=answer,
        metrics_text=metrics_text,
        rubric_text=RUBRIC_TEXT,
    )

    return JUDGE_SYSTEM, user_content


def extract_json_from_text(text: str) -> Any:
    """Extract JSON from text using multiple strategies."""
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\n(.+?)```", t, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    start = t.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == '{':
                depth += 1
            elif t[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = t[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
    m2 = re.search(r"(\{(?:.|\n)*?\})", t)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass
    return None


def validate_judge_output(parsed: dict) -> Tuple[bool, str]:
    """Validate judge output has all required fields."""
    if not isinstance(parsed, dict):
        return False, "not a JSON object"
    if 'explanation' not in parsed or 'scores' not in parsed:
        return False, "missing keys 'explanation' or 'scores'"
    scores = parsed['scores']
    if not isinstance(scores, dict):
        return False, "scores is not an object"
    for m in METRICS:
        if m not in scores:
            return False, f"missing metric {m}"
        v = scores[m]
        if not isinstance(v, int):
            return False, f"metric {m} is not int"
        if not (1 <= v <= 10):
            return False, f"metric {m} out of range"
    return True, "ok"


def judge_with_retry(judge_fn, system_msg: str, user_msg: str, retries: int = 2):
    """Call judge with retry logic for malformed responses."""
    for attempt in range(retries):
        raw, prompt_tokens, completion_tokens = judge_fn(system_msg, user_msg)
        parsed = extract_json_from_text(raw)
        ok, reason = validate_judge_output(parsed) if parsed is not None else (False, 'parse failed')
        if ok:
            return parsed, raw, prompt_tokens or 0, completion_tokens or 0
        user_msg = user_msg + "\n\nFormat-only reminder: After your brief explanation, return ONLY the compact JSON object with keys 'explanation' and 'scores' (scores must map metric names to integers 1-10)."
    return parsed, raw, prompt_tokens or 0, completion_tokens or 0


def judge_gpt4(system_msg: str, user_msg: str, temperature: float = 0.0, top_p: float = 1.0):
    """Call GPT-4 judge."""
    if ChatOpenAI is None:
        raise RuntimeError("ChatOpenAI not available for gpt4 judge")
    model = ChatOpenAI(model="gpt-4", temperature=temperature, top_p=top_p)
    prompt = system_msg + "\n\n" + user_msg
    resp = model.invoke(prompt)
    text = resp.content if hasattr(resp, 'content') else str(resp)
    return text, 0, 0


def aggregate_and_write_csv(rows: List[Dict[str, Any]], out_csv: str):
    """Generate aggregated CSV with statistics."""
    import csv
    per_judge = {}
    for r in rows:
        j = r['judge']
        per_judge.setdefault(j, []).append(r)

    fieldnames = ['judge'] + METRICS + ['overall_mean']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for j, items in per_judge.items():
            metric_vals = {m: [] for m in METRICS}
            overall_vals = []
            for it in items:
                sc = it['scores']
                vals = [sc[m] for m in METRICS]
                for m, v in zip(METRICS, vals):
                    metric_vals[m].append(v)
                overall_vals.append(sum(vals)/len(vals))
            row = [j]
            for m in METRICS:
                mean = statistics.mean(metric_vals[m]) if metric_vals[m] else 0
                std = statistics.pstdev(metric_vals[m]) if metric_vals[m] else 0
                row.append(f"{mean:.3f} (std={std:.3f})")
            row.append(f"{statistics.mean(overall_vals):.3f}" if overall_vals else "0")
            writer.writerow(row)

        all_items = rows
        metric_vals = {m: [] for m in METRICS}
        overall_vals = []
        for it in all_items:
            sc = it['scores']
            vals = [sc[m] for m in METRICS]
            for m, v in zip(METRICS, vals):
                metric_vals[m].append(v)
            overall_vals.append(sum(vals)/len(vals))
        row = ['combined']
        for m in METRICS:
            mean = statistics.mean(metric_vals[m]) if metric_vals[m] else 0
            std = statistics.pstdev(metric_vals[m]) if metric_vals[m] else 0
            row.append(f"{mean:.3f} (std={std:.3f})")
        row.append(f"{statistics.mean(overall_vals):.3f}" if overall_vals else "0")
        writer.writerow(row)


def run_evaluation_official(questions_jsonl: str, output_jsonl: str, model_name: str = "mental-health-pipeline",
                            temperature_judge: float = 0.0, seed: int = 42, verbose: bool = True):
    """Official evaluation using a standardized test set with GPT-4 judge.

    Args:
        questions_jsonl: Path to JSONL file with questions (typically 200 questions)
        output_jsonl: Path to save results
        model_name: Name of the model being evaluated
        temperature_judge: Temperature for judge (0.0 for deterministic)
        seed: Random seed for shuffling
        verbose: Print progress updates
    """
    if mental_health_pipeline_detailed is None or clear_memory is None:
        raise RuntimeError('Pipeline functions not available')

    qs = load_questions_from_jsonl(questions_jsonl)
    rng = random.Random(seed)
    rng.shuffle(qs)

    cache_path = answers_cache_path(output_jsonl)
    cache = load_cache(cache_path)

    results = []
    jsonl_fp = open(output_jsonl, 'w', encoding='utf-8')

    total_questions = len(qs)
    if verbose:
        print(f"\n{'='*80}")
        print(f"OFFICIAL EVALUATION: {total_questions} questions")
        print(f"{'='*80}\n")

    for idx, q in enumerate(qs):
        qid = q.get('question_id') or q.get('id') or str(idx)
        question = q.get('question') or q.get('text')

        if verbose:
            print(f"\n{'='*80}")
            print(f"QUESTION {idx + 1}/{total_questions} (ID: {qid})")
            print(f"{'='*80}")
            print(f"Question: {question}")
            print(f"{'-'*80}")

        if qid in cache:
            answer = cache[qid]
            if verbose:
                print(f"[Using cached answer]")
        else:
            if verbose:
                print(f"[Clearing memory for new person...]")
            clear_memory()

            if verbose:
                print(f"[Generating pipeline response...]")
                print(f"  Step 1: Clinical transformation and memory extraction...")
            pipeline_out = mental_health_pipeline_detailed(question)
            answer = pipeline_out['final_response']
            if verbose:
                print(f"  Step 2: Professional response generation...")
                print(f"  Step 3: Compassionate tone transformation...")
                print(f"[Pipeline complete]")

            cache[qid] = answer
            save_cache(cache_path, cache)

        if verbose:
            print(f"\nFinal Response:\n{answer}")
            print(f"{'-'*80}")
            print(f"[Evaluating with GPT-4 judge...]")

        system_msg, user_msg = build_prompt_for_judge(question, answer)
        parsed, raw, ptoks, ctoks = judge_with_retry(
            lambda s, u: judge_gpt4(s, u, temperature=temperature_judge, top_p=1.0),
            system_msg, user_msg)

        if parsed is None:
            if verbose:
                print(f"[WARNING: Judge evaluation failed]")
            row = {
                'question_id': qid,
                'model': model_name,
                'judge': 'gpt4-turbo',
                'scores': {m: 0 for m in METRICS},
                'explanation': '',
                'raw_prompt_tokens': ptoks,
                'raw_completion_tokens': ctoks,
            }
        else:
            scores_list = [parsed['scores'][m] for m in METRICS]
            avg_score = sum(scores_list) / len(scores_list)
            if verbose:
                print(f"[Evaluation complete - Average Score: {avg_score:.2f}/10]")
                print(f"Scores: {parsed['scores']}")

            row = {
                'question_id': qid,
                'model': model_name,
                'judge': 'gpt4-turbo',
                'scores': parsed['scores'],
                'explanation': parsed.get('explanation', ''),
                'raw_prompt_tokens': ptoks,
                'raw_completion_tokens': ctoks,
            }

        dump_jsonl_line(jsonl_fp, row)
        results.append(row)

        if verbose:
            print(f"{'='*80}\n")

    jsonl_fp.close()

    if verbose:
        print(f"\n{'='*80}")
        print(f"EVALUATION COMPLETE - Generating summary...")
        print(f"{'='*80}\n")

    out_csv = Path(output_jsonl).with_suffix('.aggregated.csv')
    aggregate_and_write_csv(results, str(out_csv))

    if verbose:
        print(f'Wrote results to {output_jsonl}')
        print(f'Wrote aggregated CSV to {out_csv}')

    return str(out_csv)


def run_evaluation_unofficial(num_samples: int = 20, output_jsonl: str = "eval_unofficial.jsonl",
                              model_name: str = "mental-health-pipeline", seed: int = 42,
                              verbose: bool = True):
    """Unofficial evaluation using random samples from the dataset.

    Quick evaluation for testing purposes. Randomly samples from MentalChat16K dataset.

    Args:
        num_samples: Number of samples to evaluate
        output_jsonl: Path to save results
        model_name: Name of the model being evaluated
        seed: Random seed for reproducibility
        verbose: Print progress updates
    """
    # Generate temporary question file
    temp_questions = f"temp_eval_{num_samples}_questions.jsonl"
    generate_test_set(num_samples, temp_questions, seed=seed)

    # Run evaluation using the official function
    result = run_evaluation_official(
        questions_jsonl=temp_questions,
        output_jsonl=output_jsonl,
        model_name=model_name,
        temperature_judge=0.0,
        seed=seed,
        verbose=verbose
    )

    # Clean up temp file
    Path(temp_questions).unlink(missing_ok=True)

    return result


def main():
    parser = argparse.ArgumentParser(description='Mental Health Pipeline Evaluation')
    parser.add_argument('--mode', choices=['official', 'unofficial'],
                       help='Evaluation mode: official (200 test set) or unofficial (random sampling)')
    parser.add_argument('--questions_jsonl', type=str,
                       help='Path to questions JSONL file (required for official mode)')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples for unofficial mode (default: 20)')
    parser.add_argument('--output', type=str, default='evaluation_results.jsonl',
                       help='Output JSONL file path')
    parser.add_argument('--model_name', type=str, default='mental-health-pipeline',
                       help='Model name for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--generate_test_set', type=int, metavar='N',
                       help='Generate a test set with N questions and exit')

    args = parser.parse_args()

    # Generate test set mode
    if args.generate_test_set:
        output_path = f"test_set_{args.generate_test_set}.jsonl"
        generate_test_set(args.generate_test_set, output_path, seed=args.seed)
        return

    # Validate mode is provided for evaluation
    if not args.mode:
        parser.error("--mode is required when not using --generate_test_set")

    # Run evaluation
    if args.mode == 'official':
        if not args.questions_jsonl:
            print("Error: --questions_jsonl is required for official mode")
            sys.exit(1)
        run_evaluation_official(
            questions_jsonl=args.questions_jsonl,
            output_jsonl=args.output,
            model_name=args.model_name,
            seed=args.seed,
            verbose=True
        )
    else:  # unofficial
        run_evaluation_unofficial(
            num_samples=args.num_samples,
            output_jsonl=args.output,
            model_name=args.model_name,
            seed=args.seed,
            verbose=True
        )


if __name__ == '__main__':
    main()
