#!/usr/bin/env python3
"""Command-line evaluation runner matching MentalChat16K judge setup.

Usage example:
python eval.py --questions_jsonl eval200.jsonl --model_name_or_endpoint my-model --judges gpt4,gemini --out results.jsonl --batch_size 8 --max_tokens 512 --seed 42

This script:
- Loads questions from a JSONL file (each line: {"question_id":..., "question":...}).
- Calls the answering pipeline (mental_health_pipeline_detailed) to get a single response per question (cached).
- For each requested judge (gpt4, gemini), calls the judge with the exact paper prompt to produce a JSON score object.
- Validates and writes results as one JSONL row per (question, model, judge).
- Emits an aggregated CSV with per-metric mean/std and overall means per judge and combined.

Notes:
- Judge temperature is fixed to 0 and top_p=1.
- The answering call uses the existing pipeline function unchanged except for caching.
"""
import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

try:
    from pipeline import mental_health_pipeline_detailed, clear_memory
except Exception:
    mental_health_pipeline_detailed = None
    clear_memory = None

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

import re


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
    "1: The response completely fails to address the metric, showing a total disregard for the user’s needs or concerns in this area.\n"
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
    "• The ratings don’t necessarily need to be the same.\n\n"
    "**User Question**\n"
    "{question}\n\n"
    "**Model Response**\n"
    "{answer}\n\n"
    "**Consultation Metrics**\n"
    "{metrics_text}\n\n"
    "**Scoring Rubrics**\n"
    "{rubric_text}\n\n"
)


def load_questions(path: str) -> List[Dict[str, Any]]:
    qs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            qs.append(json.loads(line))
    return qs


def dump_jsonl_line(fp, obj):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def answers_cache_path(out_path: str) -> str:
    return out_path + ".answers_cache.json"


def load_cache(path: str) -> Dict[str, str]:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(path: str, cache: Dict[str, str]):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def build_prompt_for_judge(question: str, answer: str) -> Tuple[str, str]:
    # Build metrics_text exactly as required
    metrics_text = (
        "Active Listening — reflects understanding without assumptions.\n"
        "Empathy & Validation — conveys understanding and validates feelings.\n"
        "Safety & Trustworthiness — avoids harm; information is consistent/reliable.\n"
        "Open-mindedness & Non-judgment — unbiased, respectful, unconditional positive regard.\n"
        "Clarity & Encouragement — clear, concise, motivating while neutral.\n"
        "Boundaries & Ethical — clarifies role/limits; suggests professional help appropriately.\n"
        "Holistic Approach — addresses emotional/cognitive/situational context broadly.\n"
    )

    rubric_text = RUBRIC_TEXT

    user_content = USER_TEMPLATE.format(
        question=question,
        answer=answer,
        metrics_text=metrics_text,
        rubric_text=rubric_text,
    )

    return JUDGE_SYSTEM, user_content


def extract_json_from_text(text: str) -> Any:
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
    # Returns (parsed_json, raw_text, prompt_tokens, completion_tokens)
    for attempt in range(retries):
        raw, prompt_tokens, completion_tokens = judge_fn(system_msg, user_msg)
        parsed = extract_json_from_text(raw)
        ok, reason = validate_judge_output(parsed) if parsed is not None else (False, 'parse failed')
        if ok:
            return parsed, raw, prompt_tokens or 0, completion_tokens or 0
        # re-ask with format-only reminder
        user_msg = user_msg + "\n\nFormat-only reminder: After your brief explanation, return ONLY the compact JSON object with keys 'explanation' and 'scores' (scores must map metric names to integers 1-10)."
    # final attempt return whatever we have
    return parsed, raw, prompt_tokens or 0, completion_tokens or 0


def judge_gpt4(system_msg: str, user_msg: str, temperature: float = 0.0, top_p: float = 1.0):
    """Call GPT-4 via LangChain ChatOpenAI if available."""
    if ChatOpenAI is None:
        raise RuntimeError("ChatOpenAI not available for gpt4 judge")
    model = ChatOpenAI(model="gpt-4", temperature=temperature, top_p=top_p)
    # use prompt template simple concatenation
    prompt = system_msg + "\n\n" + user_msg
    # Run the model
    resp = model.call_as_llm(prompt) if hasattr(model, 'call_as_llm') else model.generate([prompt])
    # Try to extract text
    if hasattr(resp, 'generations'):
        text = resp.generations[0][0].text
        # token accounting not available; set 0
        return text, 0, 0
    if isinstance(resp, str):
        return resp, 0, 0
    # fallback
    text = getattr(resp, 'text', str(resp))
    return text, 0, 0


def judge_gemini(system_msg: str, user_msg: str, temperature: float = 0.0, top_p: float = 1.0):
    """Light wrapper for Gemini Pro — if Google client not present, fall back to gpt4 wrapper.
    The wrapper returns (raw_text, prompt_tokens, completion_tokens).
    """
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        # fallback to gpt4 wrapper if Gemini client not available
        return judge_gpt4(system_msg, user_msg, temperature=temperature, top_p=top_p)
    # If genai is present, use it (this code assumes genai is configured)
    prompt = system_msg + "\n\n" + user_msg
    resp = genai.generate(model="gemini-pro", prompt=prompt)
    text = getattr(resp, 'candidates', [{}])[0].get('content', '') if resp else ''
    # token counts not available reliably; return 0
    return text, 0, 0


def aggregate_and_write_csv(rows: List[Dict[str, Any]], out_csv: str):
    # rows: each contains question_id, model, judge, scores dict
    import csv
    # compute per judge metrics
    per_judge = {}
    for r in rows:
        j = r['judge']
        per_judge.setdefault(j, []).append(r)

    fieldnames = ['judge'] + METRICS + ['overall_mean']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        # per judge
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

        # combined
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


def run_evaluation(args):
    qs = load_questions(args.questions_jsonl)
    # shuffle with seed
    rng = random.Random(args.seed)
    rng.shuffle(qs)

    cache_path = answers_cache_path(args.out)
    cache = load_cache(cache_path)

    results = []
    jsonl_fp = open(args.out, 'w', encoding='utf-8')

    total_questions = len(qs)
    print(f"\n{'='*80}")
    print(f"STARTING EVALUATION: {total_questions} questions")
    print(f"{'='*80}\n")

    for idx, q in enumerate(qs):
        qid = q.get('question_id') or q.get('id') or str(idx)
        question = q.get('question') or q.get('text')

        print(f"\n{'='*80}")
        print(f"QUESTION {idx + 1}/{total_questions} (ID: {qid})")
        print(f"{'='*80}")
        print(f"Question: {question}")
        print(f"{'-'*80}")

        if qid in cache:
            answer = cache[qid]
            print(f"[Using cached answer]")
        else:
            if mental_health_pipeline_detailed is None:
                raise RuntimeError('Answering pipeline not available in this environment')
            if clear_memory is None:
                raise RuntimeError('clear_memory function not available')

            # Clear memory for new person
            print(f"[Clearing memory for new person...]")
            clear_memory()

            # Generate response
            print(f"[Generating pipeline response...]")
            print(f"  Step 1: Clinical transformation and memory extraction...")
            pipeline_out = mental_health_pipeline_detailed(question)
            answer = pipeline_out['final_response']
            print(f"  Step 2: Professional response generation...")
            print(f"  Step 3: Compassionate tone transformation...")
            print(f"[Pipeline complete]")

            cache[qid] = answer
            save_cache(cache_path, cache)

        print(f"\nFinal Response:\n{answer}")
        print(f"{'-'*80}")

        for judge in args.judges.split(','):
            judge = judge.strip().lower()
            print(f"[Evaluating with {judge} judge...]")

            system_msg, user_msg = build_prompt_for_judge(question, answer)
            # call appropriate judge
            if judge in ('gpt4', 'gpt4-turbo', 'gpt4-turbo-instruct'):
                parsed, raw, ptoks, ctoks = judge_with_retry(
                    lambda s, u: judge_gpt4(s, u, temperature=args.temperature_judge, top_p=1.0), system_msg, user_msg)
                judge_name = 'gpt4-turbo'
            elif judge in ('gemini', 'gemini-pro', 'gemini_pro'):
                parsed, raw, ptoks, ctoks = judge_with_retry(
                    lambda s, u: judge_gemini(s, u, temperature=args.temperature_judge, top_p=1.0), system_msg, user_msg)
                judge_name = 'gemini-pro'
            else:
                raise ValueError(f'Unknown judge: {judge}')

            if parsed is None:
                # write failure row
                print(f"[WARNING: Judge evaluation failed]")
                row = {
                    'question_id': qid,
                    'model': args.model_name_or_endpoint,
                    'judge': judge_name,
                    'scores': {m: 0 for m in METRICS},
                    'explanation': '',
                    'raw_prompt_tokens': ptoks,
                    'raw_completion_tokens': ctoks,
                }
            else:
                # Calculate average score
                scores_list = [parsed['scores'][m] for m in METRICS]
                avg_score = sum(scores_list) / len(scores_list)
                print(f"[Evaluation complete - Average Score: {avg_score:.2f}/10]")
                print(f"Scores: {parsed['scores']}")

                row = {
                    'question_id': qid,
                    'model': args.model_name_or_endpoint,
                    'judge': judge_name,
                    'scores': parsed['scores'],
                    'explanation': parsed.get('explanation', ''),
                    'raw_prompt_tokens': ptoks,
                    'raw_completion_tokens': ctoks,
                }

            dump_jsonl_line(jsonl_fp, row)
            results.append(row)

        print(f"{'='*80}\n")

    jsonl_fp.close()

    # aggregated CSV
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE - Generating summary...")
    print(f"{'='*80}\n")

    out_csv = Path(args.out).with_suffix('.aggregated.csv')
    aggregate_and_write_csv(results, str(out_csv))
    print(f'Wrote results to {args.out} and aggregated CSV to {out_csv}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--questions_jsonl', required=True)
    p.add_argument('--model_name_or_endpoint', required=True)
    p.add_argument('--judges', required=True, help='comma-separated: gpt4,gemini')
    p.add_argument('--out', required=True)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_tokens', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--temperature_judge', type=float, default=0.0)
    args = p.parse_args()
    run_evaluation(args)


if __name__ == '__main__':
    main()
