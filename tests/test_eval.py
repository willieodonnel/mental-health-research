import json
import os
from pathlib import Path

import pytest

import eval as EVAL


def dummy_pipeline(question):
    return {"final_response": "This is a dummy answer to: " + question}


def dummy_judge(system, user):
    # returns a perfect parsed JSON
    obj = {
        "explanation": "Dummy explanation",
        "scores": {m: 8 for m in EVAL.METRICS}
    }
    return json.dumps(obj), 0, 0


def test_run_with_dummy(monkeypatch, tmp_path):
    # monkeypatch pipeline and judges
    monkeypatch.setattr(EVAL, 'mental_health_pipeline_detailed', lambda q: {"final_response": "ans:" + q})
    monkeypatch.setattr(EVAL, 'judge_gpt4', lambda s, u, temperature, top_p: dummy_judge(s, u))
    monkeypatch.setattr(EVAL, 'judge_gemini', lambda s, u, temperature, top_p: dummy_judge(s, u))

    questions_file = tmp_path / 'qs.jsonl'
    questions_file.write_text((Path(__file__).parent / 'questions_fixture.jsonl').read_text())

    out = tmp_path / 'results.jsonl'
    args = type('A', (), {
        'questions_jsonl': str(questions_file),
        'model_name_or_endpoint': 'dummy-model',
        'judges': 'gpt4',
        'out': str(out),
        'batch_size': 2,
        'max_tokens': 512,
        'seed': 42,
        'temperature_judge': 0.0
    })

    EVAL.run_evaluation(args)

    # read results
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2

    data = [json.loads(l) for l in lines]
    assert all('scores' in d and len(d['scores']) == len(EVAL.METRICS) for d in data)
