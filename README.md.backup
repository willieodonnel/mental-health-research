# Mental Health Support Pipeline

A local mental health inference system using **Mistral-7B-Instruct** on RTX 5080 GPU. Generates empathetic, professional counseling responses entirely on your local hardware.

**IMPORTANT**:
- ❌ Does **NOT** use ChatGPT/GPT-4 for inference
- ✅ Uses **Mistral-7B-Instruct** running locally on RTX 5080
- ✅ GPT-4 is **ONLY** used optionally as a judge for evaluation (NOT for generating responses)
- ✅ 100% private - all inference runs on your hardware

## Overview

This pipeline uses **Mistral-7B-Instruct** for single-stage local inference:

**User Input** → **Mistral-7B-Instruct (RTX 5080)** → **Mental Health Counseling Response**

### Legacy Three-Stage Pipeline (Optional)

The repository also contains a legacy three-stage LangChain pipeline using GPT-4 (in `pipeline.py`), but this is **NOT the primary inference method**:

1. **Clinical Transformation (LLM 1)**: Converts first-person user input into third-person clinical notes (GPT-4)
2. **Professional Response (LLM 2)**: Generates evidence-based responses using transformed text (GPT-4)
3. **Compassionate Communication (LLM 3)**: Transforms clinical language into warm responses (GPT-4)

**The main method is local Mistral-7B inference, NOT the GPT-4 pipeline.**

## Features

- **Local Mistral-7B Inference**: Runs entirely on RTX 5080 GPU (NO ChatGPT/GPT-4 for inference)
- **PyTorch Nightly Optimized**: Full support for RTX 5080 Blackwell architecture
- **GPU Accelerated**: 40-60 tokens/second with 4-bit quantization
- **Standardized Evaluation**: Optional GPT-4 as judge using 7 metrics from MentalChat16K research
- **Batch Processing**: Process multiple questions and save results to JSONL
- **Interactive Mode**: Real-time counseling chat interface
- **100% Private**: All inference on local hardware, no API calls for inference

## Performance

Recent improvements have significantly enhanced the pipeline:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Active Listening | 6.60 | ~9.00 | +36% |
| Holistic Approach | 6.45 | ~9.00 | +40% |
| Boundaries & Ethical | 7.05 | ~9.00 | +28% |
| **Overall Score** | **7.50** | **~9.00** | **+20%** |

One test sample achieved a **perfect 10/10 score across all 7 metrics**.

## Dataset

Uses the [MentalChat16K](https://huggingface.co/datasets/ShenLab/MentalChat16K) dataset from Hugging Face, containing 16,084 mental health conversation examples.

## Installation

### Prerequisites
- Python 3.8+
- **RTX 5080 GPU** (or compatible NVIDIA GPU with 8GB+ VRAM)
- **PyTorch Nightly** (required for RTX 5080 Blackwell architecture)
- **Optional**: OpenAI API key (ONLY for evaluation with GPT-4 judge, NOT for inference)

### Main Installation (Mistral-7B Local Inference)

```bash
# Step 1: Install PyTorch Nightly (REQUIRED for RTX 5080)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Step 2: Install inference dependencies
pip install transformers accelerate bitsandbytes sentencepiece datasets tqdm

# Step 3: Test setup
python test_setup.py

# Optional: For evaluation only (GPT-4 as judge)
pip install langchain langchain-openai python-dotenv openai
echo "OPENAI_API_KEY=your-api-key-here" > .env  # Only needed for evaluation
```

## Quick Start

### Main Usage (Mistral-7B on RTX 5080)

**No API key needed for inference!**

```bash
# Quick demo (process 3 example questions with Mistral-7B)
python mental_health_inference.py

# Custom questions
python run_custom_questions.py

# Interactive chat mode
python -c "from mental_health_inference import MentalHealthInferencePipeline; MentalHealthInferencePipeline().interactive_mode()"
```

### Evaluation (Optional - Requires OpenAI API Key)

**Mistral generates responses, GPT-4 judges them:**

```bash
# Generate test set
python evaluation_local.py --generate_test_set 20

# Run evaluation (Mistral inference + GPT-4 judge)
python evaluation_local.py --questions test_set_20.jsonl --num_samples 20
```

### Legacy GPT-4 Pipeline (Not Recommended)

```bash
# This uses GPT-4 for inference (expensive, not the main method)
python main.py
python evaluation.py --mode unofficial --num_samples 20
```

## Usage

### Main Method: Local Mistral-7B Inference (Recommended)

```python
from mental_health_inference import MentalHealthInferencePipeline

# Initialize Mistral-7B on RTX 5080 (no API key needed)
pipeline = MentalHealthInferencePipeline()

# Generate response using Mistral (NOT ChatGPT/GPT-4)
result = pipeline.generate_response(
    "How can I manage anxiety?",
    max_new_tokens=512,
    temperature=0.7
)

print(result['response'])
print(f"Speed: {result['metadata']['tokens_per_second']:.1f} tokens/sec")
```

### Legacy: GPT-4 Three-Stage Pipeline (Optional)

```python
from pipeline import mental_health_pipeline

# Uses GPT-4 (requires API key, expensive, not main method)
user_input = "I've been feeling anxious and forgetting things lately."
response = mental_health_pipeline(user_input)
print(response)
```

## Pipeline Architecture

### Main: Single-Stage Mistral-7B Inference

**User Input** → **Mistral-7B-Instruct (RTX 5080)** → **Response**

**File**: [mental_health_inference.py](mental_health_inference.py)

The model uses a single system prompt that incorporates counseling best practices:
- Empathetic and helpful approach
- Comprehensive and appropriate answers
- Evidence-based guidance
- Active listening and validation

**Key Features**:
- 100% local inference on RTX 5080
- No API calls required
- Fast generation (40-60 tokens/second)
- Fully private
- PyTorch nightly optimized for Blackwell architecture

### Legacy: Three-Stage GPT-4 Pipeline (Optional)

**File**: [pipeline.py](pipeline.py) - Uses GPT-4, not recommended as primary method

#### Stage 1: Clinical Transformation & Memory Extraction
**Function**: `llm1_transform_and_extract(user_input)` (GPT-4)
- Transforms user input to third-person clinical notes
- Extracts important facts about the patient
- Stores extracted information in memory.txt

#### Stage 2: Professional Response Generation
**Function**: `llm2_generate_professional_response(transformed_text)` (GPT-4)
- Reads patient history from memory
- Explicitly acknowledges and reflects patient's specific concerns
- Provides evidence-based clinical guidance

#### Stage 3: Compassionate Tone Transformation
**Function**: `llm3_warmify_response(professional_response)` (GPT-4)
- Transforms clinical language to warm, empathetic communication
- Preserves all specific references to patient's concerns

## Evaluation

The pipeline is evaluated using 7 standardized metrics from MentalChat16K research, with GPT-4 as the judge:

1. **Active Listening** - Reflects understanding without assumptions
2. **Empathy & Validation** - Conveys understanding and validates feelings
3. **Safety & Trustworthiness** - Avoids harm; information is consistent/reliable
4. **Open-mindedness & Non-judgment** - Unbiased, respectful, unconditional positive regard
5. **Clarity & Encouragement** - Clear, concise, motivating while neutral
6. **Boundaries & Ethical** - Clarifies role/limits; suggests professional help appropriately
7. **Holistic Approach** - Addresses emotional/cognitive/situational context broadly

### Running Evaluation

```bash
# Quick test (unofficial mode)
python evaluation.py --mode unofficial --num_samples 20

# Standardized test (official mode)
python evaluation.py --generate_test_set 200
python evaluation.py --mode official --questions_jsonl test_set_200.jsonl

# Local inference evaluation
python evaluation_local.py --num_samples 20
```

Results are saved as:
- JSONL file with detailed scores for each question
- Aggregated CSV with mean and standard deviation per metric

## File Structure

```
Mental Health Research/
├── main.py                       # Demo script (cloud mode)
├── pipeline.py                   # Three-stage pipeline implementation
├── evaluation.py                 # Evaluation script (cloud mode)
├── mental_health_inference.py    # Local inference with Mistral-7B
├── mental_health_inference_cpu.py # CPU-only inference (slow)
├── evaluation_local.py           # Evaluation for local inference
├── run_custom_questions.py       # Custom question processing
├── test_setup.py                 # Test GPU setup
├── requirements_inference.txt    # Dependencies for local inference
├── .env                          # API keys (DO NOT COMMIT)
├── .gitignore                    # Git ignore rules
├── memory.txt                    # Patient memory storage (auto-created, ignored)
└── README.md                     # This file
```

## Configuration

### Cloud Mode (OpenAI)

#### Modify LLM Model

```python
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
# Change to: model="gpt-3.5-turbo" for faster/cheaper responses
```

#### Adjust Temperature

Different temperatures are used for each stage:
- **LLM1**: 0.7 (moderate creativity for natural transformation)
- **LLM2**: 0.5 (lower for consistent, evidence-based responses)
- **LLM3**: 0.7 (moderate for natural warmth)

### Local Inference Mode

#### Memory Usage

| Quantization | VRAM Used | Quality | Speed |
|--------------|-----------|---------|-------|
| 4-bit (NF4) | ~3.5 GB | Good | Fast |
| 8-bit | ~7 GB | Better | Medium |
| 16-bit (BF16) | ~14 GB | Best | Slower |

Default: 4-bit (optimal for RTX 5080)

#### Generation Parameters

```python
result = pipeline.generate_response(
    question,
    max_new_tokens=512,    # Response length
    temperature=0.7,       # Creativity (0.1=focused, 1.0=creative)
    top_p=0.9,            # Nucleus sampling
    top_k=50              # Top-k sampling
)
```

#### Performance (RTX 5080)
- **Load Time**: 30-60 seconds (first run)
- **VRAM Usage**: 3.5-4GB
- **Speed**: 40-60 tokens/second
- **Response Time**: 2-4 seconds (for ~200 tokens)

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
concentration and short-term memory...
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

## Recent Improvements

The pipeline has been significantly enhanced through prompt engineering:

### Active Listening (+36% improvement)
- Added explicit instruction to acknowledge and reflect specific concerns
- Starts responses by addressing what the patient shared
- Demonstrates understanding before providing guidance

### Holistic Approach (+40% improvement)
- Added comprehensive checklist covering 5 dimensions
- Ensures responses address emotional, cognitive, behavioral, situational, and physical aspects
- Integrates multiple perspectives naturally

### Boundaries & Ethical (+28% improvement)
- Clarifies role and limits of text-based support
- Consistently suggests professional help when appropriate
- Removed formal signatures/sign-offs for more natural communication

See [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md) for detailed analysis.

## Troubleshooting

### Cloud Mode
- **API Key Issues**: Ensure `.env` file exists with valid `OPENAI_API_KEY`
- **Rate Limits**: Add delays between requests or reduce batch size
- **Token Costs**: Each evaluation uses significant tokens (3 LLM calls + judge)

### Local Inference Mode
- **CUDA out of memory**: Already using 4-bit quantization (most efficient)
- **Model downloading slowly**: First run downloads ~4GB, be patient
- **Module not found**: `pip install transformers accelerate bitsandbytes`
- **Slow on CPU**: Use [mental_health_inference_cpu.py](mental_health_inference_cpu.py) (very slow)

## Important Notes

**Disclaimer**: This tool is for research and educational purposes only. It is not a substitute for professional mental health care.

- Always consult qualified mental health professionals for clinical decisions
- Patient privacy and data security should be prioritized in production use
- Ensure compliance with healthcare regulations (HIPAA, GDPR, etc.)

## Dependencies

### Cloud Mode
- `datasets` - Hugging Face datasets library
- `langchain` - LangChain framework
- `langchain-openai` - OpenAI integration for LangChain
- `langchain-community` - Community integrations
- `python-dotenv` - Environment variable management
- `openai` - OpenAI API client

### Local Inference Mode
- `torch` - PyTorch with CUDA support
- `transformers` - Hugging Face transformers library
- `accelerate` - Model acceleration utilities
- `bitsandbytes` - Quantization library
- `sentencepiece` - Tokenizer support

## License

This project is for research and educational purposes.

## Contributing

This is a research project. Suggestions and improvements are welcome.

## Citation

Based on the evaluation methodology from:
- **MentalChat16K**: A Benchmark Dataset for Conversational Mental Health Assistance (ICLR 2025 submission)
