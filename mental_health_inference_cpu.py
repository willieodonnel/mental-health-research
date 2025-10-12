"""
Mental Health Inference Pipeline using Mistral-7B-Instruct (CPU-Compatible Version)

This version runs on CPU without quantization to bypass RTX 5080 sm_120 compatibility issues.
Will be slower but functional while we wait for full CUDA support.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
from pathlib import Path


class MentalHealthInferencePipeline:
    """
    CPU-compatible inference pipeline for mental health counseling.
    """

    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", use_half_precision=True):
        """
        Initialize the pipeline.

        Args:
            model_name: HuggingFace model identifier
            use_half_precision: Use FP16 to save memory (recommended)
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing Mental Health Inference Pipeline (CPU Mode)...")
        print(f"Model: {model_name}")
        print("NOTE: Running on CPU - this will be slower than GPU but functional")

        self.model_name = model_name
        self.device = "cpu"

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading tokenizer...")
        self.tokenizer = AutoModelForCausalLM.from_pretrained(model_name)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading model...")
        print("This may take 2-3 minutes on first run...")

        # Load model on CPU with FP32 (full precision for compatibility)
        self.model = AutoModel ForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Full precision for CPU
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Optimize CPU memory
        )

        # Set model to evaluation mode
        self.model.eval()

        # System prompt from the paper
        self.system_prompt = (
            "You are a helpful and empathetic mental health counseling assistant. "
            "Please answer the mental health questions based on the user's description. "
            "The assistant gives helpful, comprehensive, and appropriate answers to the user's questions."
        )

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Model loaded successfully on CPU!")

    def format_prompt(self, user_question):
        """Format the prompt in Mistral's instruction format."""
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating response (this will take 30-60 seconds on CPU)...")
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

        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response
        if "[/INST]" in full_response:
            response = full_response.split("[/INST]")[-1].strip()
        else:
            response = full_response

        output_length = len(outputs[0]) - input_length
        tokens_per_second = output_length / generation_time if generation_time > 0 else 0

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Generation complete!")
        print(f"Time: {generation_time:.2f}s")
        print(f"Speed: {tokens_per_second:.1f} tokens/sec")
        print(f"Tokens: {output_length}")

        return {
            "question": question,
            "response": response,
            "metadata": {
                "model": self.model_name,
                "generation_time_seconds": generation_time,
                "tokens_generated": output_length,
                "tokens_per_second": tokens_per_second,
                "temperature": temperature,
                "timestamp": datetime.now().isoformat(),
                "device": "CPU"
            }
        }

    def process_batch(self, questions, output_file="results_cpu.jsonl", **generation_kwargs):
        """Process a batch of questions and save results to JSONL."""
        results = []
        output_path = Path(output_file)

        print(f"\n{'='*80}")
        print(f"Processing Batch of {len(questions)} Questions (CPU Mode)")
        print(f"Output will be saved to: {output_path.absolute()}")
        print(f"{'='*80}\n")

        for i, question in enumerate(questions, 1):
            print(f"\n{'─'*80}")
            print(f"Question {i}/{len(questions)}")
            print(f"{'─'*80}")

            result = self.generate_response(question, **generation_kwargs)
            results.append(result)

            # Save incrementally to JSONL
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

            print(f"\n[OK] Saved to {output_file}")

        print(f"\n{'='*80}")
        print(f"Batch Processing Complete!")
        print(f"{'='*80}\n")

        return results

    def interactive_mode(self):
        """Run interactive counseling session in terminal."""
        print(f"\n{'='*80}")
        print("Mental Health Counseling Assistant - Interactive Mode (CPU)")
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
                print(f"\nError: {e}\n")


def main():
    """Main function with example usage."""
    print("\n" + "="*80)
    print("Mental Health Inference Pipeline (CPU Mode)")
    print("Based on MentalChat16K Paper Methodology")
    print("="*80 + "\n")

    # Initialize pipeline
    pipeline = MentalHealthInferencePipeline()

    # Example question (just one for CPU speed)
    example_questions = [
        "How can I manage my anxiety when it feels overwhelming?",
    ]

    # Process batch
    print("\nRunning CPU inference on example question...\n")
    results = pipeline.process_batch(
        example_questions,
        output_file="mental_health_responses_cpu.jsonl",
        max_new_tokens=256,  # Shorter for CPU speed
        temperature=0.7
    )

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    for i, result in enumerate(results, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {result['question']}")
        print(f"A: {result['response'][:200]}...")
        print(f"Time: {result['metadata']['generation_time_seconds']:.2f}s "
              f"({result['metadata']['tokens_per_second']:.1f} tok/s)")


if __name__ == "__main__":
    main()
