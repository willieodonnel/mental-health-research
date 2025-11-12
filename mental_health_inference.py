"""
Mental Health Inference Pipeline using Mistral-7B-Instruct on RTX 5080

IMPORTANT:
- This pipeline uses ONLY Mistral-7B-Instruct for inference (NO ChatGPT/GPT-4)
- Runs 100% locally on your RTX 5080 GPU with PyTorch nightly
- GPT-4 is ONLY used as a judge for evaluation (see evaluation_local.py)

Technical Details:
- Model: Mistral-7B-Instruct-v0.2
- Precision: FP16 or 4-bit quantization
- VRAM: ~3.5GB (4-bit) to ~14GB (FP16)
- Speed: 40-60 tokens/second on RTX 5080
- PyTorch: Nightly build required for RTX 5080 Blackwell architecture
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import json
from datetime import datetime
from pathlib import Path
import os

# Set environment variable to enable compatibility mode for sm_120
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class MentalHealthInferencePipeline:
    """
    Local inference pipeline for mental health counseling using Mistral-7B-Instruct.

    This pipeline runs ENTIRELY on your RTX 5080 GPU:
    - NO ChatGPT/GPT-4 for inference
    - NO API calls required
    - 100% private and local

    Uses FP16 precision or 4-bit quantization depending on configuration.
    Optimized for RTX 5080 with PyTorch nightly (CUDA 12.8+).
    """

    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize the pipeline with FP16 model.

        Args:
            model_name: HuggingFace model identifier
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing Mental Health Inference Pipeline...")
        print(f"Model: {model_name}")

        self.model_name = model_name

        # RTX 5080 with PyTorch Nightly (CUDA 12.8+)
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU Detected: {gpu_name}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"PyTorch Version: {torch.__version__}")

            # Check if this is RTX 5080
            if "5080" in gpu_name:
                print("[OK] RTX 5080 detected with PyTorch nightly - GPU acceleration enabled!")
            else:
                print(f"[OK] GPU acceleration enabled on {gpu_name}")
        else:
            self.device = "cpu"
            print("WARNING: CUDA not available, running on CPU (will be very slow)")
            print("For RTX 5080, ensure PyTorch nightly is installed:")
            print("pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128")

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.device == "cuda":
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading model on GPU with FP16 precision...")
            print("This may take a few minutes on first run...")
            print("Note: Using eager execution mode for sm_120 compatibility...")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # FP16 for GPU
                device_map="auto",  # Auto device mapping
                trust_remote_code=True,
                attn_implementation="eager",  # Use eager mode for sm_120 compatibility
            )
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading model on CPU...")
            print("This may take a few minutes on first run...")
            print("Note: Running on CPU - generation will be slower but will work")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # FP32 for CPU
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

        # Set model to evaluation mode
        self.model.eval()

        # System prompt from the paper
        self.system_prompt = (
            "You are a helpful and empathetic mental health counseling assistant. "
            "Please answer the mental health questions based on the user's description. "
            "The assistant gives helpful, comprehensive, and appropriate answers to the user's questions."
        )

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Model loaded successfully!")
        self._print_memory_usage()

    def _print_memory_usage(self):
        """Print current GPU memory usage."""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\n[GPU Memory Usage]")
            print(f"   Allocated: {allocated:.2f} GB")
            print(f"   Reserved:  {reserved:.2f} GB\n")

    def format_prompt(self, user_question):
        """
        Format the prompt in Mistral's instruction format.

        Args:
            user_question: The user's mental health question

        Returns:
            Formatted prompt string
        """
        # Mistral Instruct format: [INST] instruction [/INST]
        prompt = f"[INST] {self.system_prompt}\n\nUser Question: {user_question} [/INST]"
        return prompt

    def generate_response(
        self,
        question,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True
    ):
        """
        Generate a counseling response to a mental health question.

        Args:
            question: User's mental health question
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (vs greedy decoding)

        Returns:
            dict with question, response, and metadata
        """
        print(f"\n{'='*80}")
        print(f"Processing Question...")
        print(f"{'='*80}\n")

        # Format prompt
        prompt = self.format_prompt(question)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        # Generate
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating response...")
        start_time = datetime.now()

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()

        # Decode response (exclude the input prompt)
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response (after [/INST])
        if "[/INST]" in full_response:
            response = full_response.split("[/INST]")[-1].strip()
        else:
            response = full_response

        output_length = len(outputs[0]) - input_length
        tokens_per_second = output_length / generation_time if generation_time > 0 else 0

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Generation complete!")
        print(f"[Time] {generation_time:.2f}s")
        print(f"[Speed] {tokens_per_second:.1f} tokens/sec")
        print(f"[Tokens] {output_length}")

        return {
            "question": question,
            "response": response,
            "metadata": {
                "model": self.model_name,
                "generation_time_seconds": generation_time,
                "tokens_generated": output_length,
                "tokens_per_second": tokens_per_second,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "timestamp": datetime.now().isoformat()
            }
        }

    def process_batch(self, questions, output_file="results.jsonl", **generation_kwargs):
        """
        Process a batch of questions and save results to JSONL.

        Args:
            questions: List of question strings
            output_file: Path to save results
            **generation_kwargs: Arguments to pass to generate_response

        Returns:
            List of result dictionaries
        """
        results = []
        output_path = Path(output_file)

        print(f"\n{'='*80}")
        print(f"Processing Batch of {len(questions)} Questions")
        print(f"Output will be saved to: {output_path.absolute()}")
        print(f"{'='*80}\n")

        for i, question in enumerate(questions, 1):
            print(f"\n{'-'*80}")
            print(f"Question {i}/{len(questions)}")
            print(f"{'-'*80}")

            result = self.generate_response(question, **generation_kwargs)
            results.append(result)

            # Save incrementally to JSONL
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

            print(f"\n[OK] Saved to {output_file}")

        print(f"\n{'='*80}")
        print(f"[COMPLETE] Batch Processing Complete!")
        print(f"{'='*80}\n")

        return results

    def interactive_mode(self):
        """
        Run interactive counseling session in terminal.
        """
        print(f"\n{'='*80}")
        print("Mental Health Counseling Assistant - Interactive Mode")
        print(f"{'='*80}")
        print("\nType your questions and press Enter. Type 'quit' or 'exit' to end.\n")

        while True:
            try:
                question = input("You: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using the Mental Health Counseling Assistant.")
                    break

                if not question:
                    continue

                result = self.generate_response(question)
                print(f"\nAssistant:\n{result['response']}\n")

            except KeyboardInterrupt:
                print("\n\nSession ended by user.")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}\n")


def main():
    """
    Main function with example usage.
    """
    print("\n" + "="*80)
    print("Mental Health Inference Pipeline")
    print("Based on MentalChat16K Paper Methodology")
    print("="*80 + "\n")

    # Initialize pipeline
    pipeline = MentalHealthInferencePipeline()

    # Example questions (from the paper's test set domain)
    example_questions = [
        "I've been feeling so overwhelmed lately. I can't seem to focus on anything, and I've been having panic attacks almost daily. I've tried to reach out to friends and family, but they don't seem to understand what I'm going through. I feel so alone and lost.",

        "I've been having trouble sleeping for the past few months. I've tried various methods, like melatonin, meditation, and even changing my bedtime routine, but nothing seems to be working. I've also been feeling increasingly anxious and irritable.",

        "How can I manage my anxiety when it feels overwhelming?",
    ]

    # Option 1: Process batch of questions
    print("\n[Running batch inference on example questions...]\n")
    results = pipeline.process_batch(
        example_questions,
        output_file="mental_health_responses.jsonl",
        max_new_tokens=512,
        temperature=0.7
    )

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    for i, result in enumerate(results, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {result['question'][:100]}...")
        print(f"A: {result['response'][:200]}...")
        print(f"[Time] {result['metadata']['generation_time_seconds']:.2f}s "
              f"({result['metadata']['tokens_per_second']:.1f} tok/s)")

    # Option 2: Interactive mode (uncomment to use)
    # print("\n\nStarting interactive mode...\n")
    # pipeline.interactive_mode()


if __name__ == "__main__":
    main()
