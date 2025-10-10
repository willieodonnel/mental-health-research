# Mental Health Support Pipeline

A three-stage LangChain pipeline that processes user mental health input through clinical transformation, professional response generation, and compassionate communication.

## Overview

This pipeline transforms user input through three sequential LLM stages:

1. **Clinical Transformation (LLM 1)**: Converts first-person user input into third-person clinical notes, extracts important facts, and monitors cognitive patterns
2. **Professional Response (LLM 2)**: Generates evidence-based, professional clinical responses using transformed text and patient history
3. **Compassionate Communication (LLM 3)**: Transforms clinical language into warm, empathetic responses while preserving factual content

## Dataset

Uses the [MentalChat16K](https://huggingface.co/datasets/ShenLab/MentalChat16K) dataset from Hugging Face, containing 16,084 mental health conversation examples.

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Install Dependencies

```bash
pip install datasets langchain langchain-openai langchain-community python-dotenv
```

### Load Dataset

The dataset is automatically loaded when running the script:

```python
from datasets import load_dataset
ds = load_dataset("ShenLab/MentalChat16K")
```

## Configuration

### Setting Up Your API Key (Recommended Method)

Create a `.env` file in the project root directory:

```
OPENAI_API_KEY=your-api-key-here
```

The `.env` file is automatically loaded by the application and is included in `.gitignore` to prevent accidental exposure of your API key on GitHub.

### Alternative: Environment Variable

You can also set your OpenAI API key as an environment variable:

```bash
# Linux/macOS
export OPENAI_API_KEY="your-api-key-here"

# Windows (Command Prompt)
set OPENAI_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```python
from main import mental_health_pipeline

user_input = "I've been feeling anxious and forgetting things lately."
response = mental_health_pipeline(user_input)
print(response)
```

### Run Example

```bash
python main.py
```

This will run the pipeline with a sample input demonstrating anxiety and memory concerns.

## Pipeline Architecture

### Stage 1: Clinical Transformation & Memory Extraction

**Function**: `llm1_transform_and_extract(user_input)`

- Transforms user input to third-person clinical notes
- Extracts important facts about the patient
- Identifies cognitive patterns and unusual mental states
- Stores extracted information in `memory.txt`

**Output**:
- Transformed clinical text
- Extracted memory entries

### Stage 2: Professional Response Generation

**Function**: `llm2_generate_professional_response(transformed_text)`

- Reads patient history from `memory.txt`
- Generates professional, evidence-based clinical response
- Considers both current observation and historical context
- Maintains factual, measured, and considerate tone

**Output**: Professional clinical response

### Stage 3: Compassionate Tone Transformation

**Function**: `llm3_warmify_response(professional_response)`

- Transforms clinical language to warm, empathetic communication
- Preserves all factual content and recommendations
- Uses accessible, supportive language
- Makes patient feel heard and cared for

**Output**: Final warm and compassionate response

## Memory System

The pipeline maintains persistent memory in `memory.txt`:

- **Automatic Storage**: Important facts are automatically extracted and stored by LLM 1
- **Context-Aware**: Subsequent responses use accumulated patient history
- **Cognitive Monitoring**: Tracks patterns related to cognition and mental state
- **Append-Only**: New memories are added without overwriting existing entries

## Evaluation

The pipeline can be evaluated using the standardized MentalChat16K metrics with GPT-4 as the judge.

### Unofficial Evaluation (Quick Testing)

For quick testing with random samples from the dataset:

```bash
python evaluation.py --mode unofficial --num_samples 20
```

### Official Evaluation (200 Test Set)

For standardized evaluation with the 200-question test set:

```bash
# First, generate the test set
python evaluation.py --generate_test_set 200

# Then run the official evaluation
python evaluation.py --mode official --questions_jsonl test_set_200.jsonl
```

### Evaluation Metrics

The evaluation uses 7 metrics from the MentalChat16K research:
1. **Active Listening** - reflects understanding without assumptions
2. **Empathy & Validation** - conveys understanding and validates feelings
3. **Safety & Trustworthiness** - avoids harm; information is consistent/reliable
4. **Open-mindedness & Non-judgment** - unbiased, respectful, unconditional positive regard
5. **Clarity & Encouragement** - clear, concise, motivating while neutral
6. **Boundaries & Ethical** - clarifies role/limits; suggests professional help appropriately
7. **Holistic Approach** - addresses emotional/cognitive/situational context broadly

Results are saved as:
- JSONL file with detailed scores for each question
- Aggregated CSV with mean and standard deviation per metric

## File Structure

```
Mental Health Research/
├── main.py           # Demo script for the pipeline
├── pipeline.py       # Three-stage pipeline implementation
├── evaluation.py     # Evaluation script (official & unofficial modes)
├── .env              # Environment variables (API key) - DO NOT COMMIT
├── .gitignore        # Git ignore rules (includes .env and memory.txt)
├── memory.txt        # Patient memory storage (auto-created)
└── README.md         # This file
```

## Example Output

### Input
```
I've been feeling really anxious lately. I can't seem to focus on anything
and I keep forgetting things. Yesterday I forgot my meeting and today I left
my keys in the door. I'm worried something is wrong with my memory.
```

### Stage 1 Output (Clinical Note)
```
The patient reports experiencing anxiety. The patient states difficulty
with concentration and recent memory issues, including forgetting a meeting
and leaving keys in the door.
```

### Stage 2 Output (Professional Response)
```
Based on the patient's reported symptoms of anxiety, attention difficulties,
and memory concerns, these experiences may be related to stress and anxiety
rather than a primary memory disorder. Anxiety can significantly impact
concentration and short-term memory. A comprehensive evaluation would be
beneficial to rule out other contributing factors.
```

### Stage 3 Output (Final Response)
```
I hear that you're feeling anxious and worried about your memory. It's
completely understandable to be concerned when you're forgetting important
things. What you're experiencing with focus and memory often happens when
we're under stress or feeling anxious - our minds can become overwhelmed.
I'd like to help you explore this further to understand what's contributing
to these difficulties and find ways to support you.
```

## Customization

### Modify LLM Model

```python
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
# Change to: model="gpt-3.5-turbo" for faster/cheaper responses
```

### Adjust Temperature

```python
llm = ChatOpenAI(model="gpt-4", temperature=0.5)
# Lower = more consistent, Higher = more creative
```

### Custom Memory File Location

```python
MEMORY_FILE = Path("custom_memory_location.txt")
```

## Important Notes

⚠️ **Disclaimer**: This tool is for research and educational purposes only. It is not a substitute for professional mental health care.

- Always consult qualified mental health professionals for clinical decisions
- Patient privacy and data security should be prioritized in production use
- Ensure compliance with healthcare regulations (HIPAA, GDPR, etc.)

## Dependencies

- `datasets` - Hugging Face datasets library
- `langchain` - LangChain framework
- `langchain-openai` - OpenAI integration for LangChain
- `langchain-community` - Community integrations for LangChain
- `python-dotenv` - Environment variable management
- `openai` - OpenAI API client

## License

This project is for research and educational purposes.

## Contributing

This is a research project. Suggestions and improvements are welcome.
