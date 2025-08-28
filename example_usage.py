#!/usr/bin/env python3
"""
Example usage script for basic_prompt_optimizer.py

This script demonstrates how to use the PromptOptimizer class to optimize
prompts for Llama and Qwen family models.
"""

from basic_prompt_optimizer import PromptOptimizer, PromptOptimizationConfig

def main():
    """Example usage of the PromptOptimizer"""
    
    # Configuration for the optimizer
    config = PromptOptimizationConfig(
        model_path="meta-llama/Llama-3.1-8B-Instruct",  # Change to your model path
        device="cuda",  # Use "cpu" if no GPU available
        torch_dtype="auto",
        num_iterations=10,
        population_size=6,
        optimization_strategy="genetic",
        max_new_tokens=128,
        temperature=0.7,
        output_dir="./optimization_results"
    )
    
    # Initialize the optimizer
    print("Initializing PromptOptimizer...")
    optimizer = PromptOptimizer(config)
    
    # Define the optimization task
    initial_prompt = "Write a creative story"
    
    target_examples = [
        "Once upon a time in a magical forest, a young wizard discovered an ancient spellbook that would change everything.",
        "The spaceship landed silently on the alien planet, and Captain Sarah knew their mission had just begun.",
        "In the bustling streets of Tokyo, Maya found a mysterious letter that led her on an incredible adventure."
    ]
    
    context = "You are a creative writing assistant focused on engaging storytelling."
    
    # Run the optimization
    print("Starting prompt optimization...")
    results = optimizer.optimize_prompt(
        initial_prompt=initial_prompt,
        target_examples=target_examples,
        objective="similarity",
        context=context
    )
    
    # Display results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Initial prompt: {results['initial_prompt']}")
    print(f"Optimized prompt: {results['optimized_prompt']}")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Total iterations: {results['total_iterations']}")
    print(f"Optimization time: {results['optimization_time']:.2f} seconds")
    
    print("\nGenerated examples with optimized prompt:")
    for i, example in enumerate(results['final_generated_examples'], 1):
        print(f"{i}. {example}")
    
    print("\nResults saved to:", config.output_dir)
    print("="*70)

if __name__ == "__main__":
    main()