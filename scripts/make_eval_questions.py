from datasets import load_dataset
import json
from pathlib import Path

OUT = Path("../eval_20_questions.jsonl")
OUT.unlink(missing_ok=True)

ds = load_dataset("ShenLab/MentalChat16K")
# Try to find the first split that contains items
split = next(iter(ds.keys()))
count = 0
with OUT.open('w', encoding='utf-8') as f:
    for item in ds[split]:
        if count >= 20:
            break
        qid = item.get('question_id') or item.get('id') or item.get('qid') or str(count)
        # try a few fields for text
        question = item.get('question') or item.get('text') or item.get('prompt') or item.get('instruction') or ''
        obj = {'question_id': qid, 'question': question}
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        count += 1
print(f'Wrote {count} questions to {OUT.resolve()}')
