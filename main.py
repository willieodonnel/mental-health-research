"""
Main script to run the mental health pipeline evaluation.

This script imports the pipeline from pipeline.py and the evaluation
functions from evaluation.py, then runs an evaluation on a specified
number of samples from the MentalChat16K dataset.
"""

from pipeline import mental_health_pipeline_detailed
from evaluation import evaluate_pipeline
from datetime import datetime

if __name__ == "__main__":
    print("=" * 60)
    print("MENTAL HEALTH PIPELINE EVALUATION")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    NUM_SAMPLES = 10
    OUTPUT_CSV = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print(f"Configuration:")
    print(f"  - Number of samples: {NUM_SAMPLES}")
    print(f"  - Output file: {OUTPUT_CSV}")
    print()

    # Run evaluation
    evaluate_pipeline(
        pipeline_function=mental_health_pipeline_detailed,
        num_samples=NUM_SAMPLES,
        output_csv=OUTPUT_CSV
    )

    print()
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
