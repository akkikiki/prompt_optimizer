#!/usr/bin/env python3
"""
Prompt Optimization Script for Llama and Qwen Family Models

This script optimizes prompts for Llama and Qwen models by:
1. Using iterative refinement with gradient-free optimization
2. Supporting various model variants and configurations
3. Implementing genetic algorithm-inspired prompt evolution
4. Using perplexity-based evaluation for prompt quality
5. Leveraging standard transformers generation capabilities

Features:
- Multi-strategy optimization (genetic, random search, hill climbing)
- Automatic model detection and configuration
- Template-aware optimization for instruction-tuned models
- Batch evaluation for efficiency
- Comprehensive scoring metrics
- Standard transformers integration
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig
)

# Import standard transformers modules
import argparse
import json
import logging
import random
import re
from pathlib import Path
from collections import defaultdict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PromptOptimizationConfig:
    """Configuration for prompt optimization with Llama/Qwen models"""
    # Model configuration
    model_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer_path: Optional[str] = None
    device: str = "cuda"
    torch_dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"
    
    # Optimization parameters
    num_iterations: int = 15
    population_size: int = 8  # Number of prompts in each generation
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    optimization_strategy: str = "genetic"  # "genetic", "random", "hill_climbing", "hybrid"
    
    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = 50
    do_sample: bool = True
    num_beams: int = 1
    
    # Evaluation parameters
    num_evaluations_per_prompt: int = 3
    perplexity_weight: float = 0.3
    similarity_weight: float = 0.4
    diversity_weight: float = 0.2
    length_weight: float = 0.1
    
    # Template configuration
    use_instruction_template: bool = True
    system_message: Optional[str] = None
    instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n"
    
    # Output configuration
    max_length: int = 2048
    output_dir: str = "./optimized_prompts"
    save_intermediate_results: bool = True
    
    # Advanced options
    early_stopping_patience: int = 3
    min_improvement_threshold: float = 0.01
    use_cache: bool = True
    seed: Optional[int] = None


class PromptOptimizer:
    """
    Optimizes prompts for Llama and Qwen family models using genetic algorithms
    and other optimization strategies.
    """
    
    def __init__(self, config: PromptOptimizationConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Set random seeds for reproducibility
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            random.seed(config.seed)
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        # Initialize optimization state
        self.optimization_history = []
        self.best_prompts = []
        self.evaluation_cache = {}
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PromptOptimizer initialized with device: {self.device}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Optimization strategy: {config.optimization_strategy}")
    
    def _load_model_and_tokenizer(self):
        """Load Llama/Qwen model and tokenizer with proper configuration"""
        try:
            # Determine torch dtype
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
                "auto": "auto"
            }
            torch_dtype = dtype_map.get(self.config.torch_dtype, "auto")
            
            # Load tokenizer
            tokenizer_path = self.config.tokenizer_path or self.config.model_path
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load Llama/Qwen model using standard transformers
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch_dtype,
                device_map="auto" if self.config.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            logger.info("Successfully loaded model")
            
            if self.config.device != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Store model name for logging
            self.model_name = getattr(self.model.config, '_name_or_path', self.config.model_path)
            
            # Setup generation config
            self.generation_config = GenerationConfig(
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=self.config.do_sample,
                num_beams=self.config.num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                #use_cache=self.config.use_cache
            )
            
            logger.info(f"Successfully loaded model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def optimize_prompt(
        self,
        initial_prompt: str,
        target_examples: List[str],
        objective: str = "similarity",
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize a prompt to better produce target examples.
        
        Args:
            initial_prompt: Starting prompt to optimize
            target_examples: List of desired outputs
            objective: Optimization objective ("similarity", "diversity", "perplexity")
            context: Optional context or system message
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info(f"Starting prompt optimization")
        logger.info(f"Initial prompt: {initial_prompt}")
        logger.info(f"Target examples: {len(target_examples)}")
        logger.info(f"Objective: {objective}")
        
        # Initialize optimization
        start_time = time.time()
        self.objective = objective
        self.target_examples = target_examples
        self.context = context
        
        # Create initial population
        if self.config.optimization_strategy == "genetic":
            population = self._create_initial_population(initial_prompt)
        else:
            population = [initial_prompt]
        
        best_prompt = initial_prompt
        best_score = float('-inf')
        no_improvement_count = 0
        
        # Main optimization loop
        for iteration in range(self.config.num_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.config.num_iterations}")
            
            # Evaluate population
            scores = self._evaluate_population(population, target_examples, context)
            
            # Find best in current population
            current_best_idx = np.argmax(scores)
            current_best_score = scores[current_best_idx]
            current_best_prompt = population[current_best_idx]
            
            # Check for improvement
            if current_best_score > best_score + self.config.min_improvement_threshold:
                best_score = current_best_score
                best_prompt = current_best_prompt
                no_improvement_count = 0
                logger.info(f"New best score: {best_score:.4f}")
            else:
                no_improvement_count += 1
            
            # Record iteration results
            iteration_result = {
                'iteration': iteration + 1,
                'population': population.copy(),
                'scores': scores.tolist(),
                'best_prompt': current_best_prompt,
                'best_score': current_best_score,
                'population_diversity': self._calculate_population_diversity(population)
            }
            self.optimization_history.append(iteration_result)
            
            # Early stopping
            if no_improvement_count >= self.config.early_stopping_patience:
                logger.info(f"Early stopping after {iteration + 1} iterations")
                break
            
            # Generate next population
            if iteration < self.config.num_iterations - 1:
                if self.config.optimization_strategy == "genetic":
                    population = self._evolve_population(population, scores)
                elif self.config.optimization_strategy == "hill_climbing":
                    population = self._hill_climbing_step(population, scores)
                elif self.config.optimization_strategy == "random":
                    population = self._random_search_step(initial_prompt)
                else:  # hybrid
                    population = self._hybrid_optimization_step(population, scores, initial_prompt)
        
        # Generate final examples
        final_examples = self._generate_examples(best_prompt, context, num_examples=5)
        
        # Compile results
        optimization_time = time.time() - start_time
        results = {
            'initial_prompt': initial_prompt,
            'optimized_prompt': best_prompt,
            'best_score': best_score,
            'target_examples': target_examples,
            'final_generated_examples': final_examples,
            'optimization_history': self.optimization_history,
            'optimization_time': optimization_time,
            'total_iterations': len(self.optimization_history),
            'objective': objective,
            'context': context,
            'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _create_initial_population(self, initial_prompt: str) -> List[str]:
        """Create initial population for genetic algorithm"""
        population = [initial_prompt]
        
        # Create variations using different strategies
        variation_strategies = [
            self._add_instruction_words,
            self._rephrase_prompt,
            self._add_context_words,
            self._modify_structure,
            self._add_examples_hint,
            self._change_tone,
            self._add_constraints
        ]
        
        while len(population) < self.config.population_size:
            strategy = random.choice(variation_strategies)
            variant = strategy(initial_prompt)
            if variant not in population:
                population.append(variant)
        
        return population[:self.config.population_size]
    
    def _add_instruction_words(self, prompt: str) -> str:
        """Add instructional words to the prompt"""
        instruction_words = [
            "Please", "Carefully", "Specifically", "Clearly", "Detailed",
            "Step by step", "Thoroughly", "Precisely", "Comprehensively"
        ]
        word = random.choice(instruction_words)
        return f"{word} {prompt.lower()}"
    
    def _rephrase_prompt(self, prompt: str) -> str:
        """Rephrase the prompt with different wording"""
        rephrase_patterns = [
            ("write", "create"),
            ("explain", "describe"),
            ("tell me", "provide"),
            ("how to", "the way to"),
            ("what is", "define"),
            ("give me", "provide me with")
        ]
        
        result = prompt
        for old, new in rephrase_patterns:
            if old in result.lower():
                result = result.lower().replace(old, new)
                break
        return result
    
    def _add_context_words(self, prompt: str) -> str:
        """Add context-setting words"""
        context_words = [
            "In detail,", "For reference,", "As an expert,", "To clarify,",
            "For example,", "In practice,", "Specifically,", "Generally,"
        ]
        word = random.choice(context_words)
        return f"{word} {prompt.lower()}"
    
    def _modify_structure(self, prompt: str) -> str:
        """Modify the structural format of the prompt"""
        if random.random() < 0.5:
            return f"Here's what I need: {prompt}"
        else:
            return f"I want you to {prompt.lower()}"
    
    def _add_examples_hint(self, prompt: str) -> str:
        """Add hints about providing examples"""
        example_hints = [
            "with examples", "including specific examples", "with concrete examples",
            "providing examples", "showing examples"
        ]
        hint = random.choice(example_hints)
        return f"{prompt} {hint}"
    
    def _change_tone(self, prompt: str) -> str:
        """Change the tone of the prompt"""
        tones = [
            ("professional", "In a professional manner,"),
            ("casual", "In simple terms,"),
            ("academic", "From an academic perspective,"),
            ("practical", "In practical terms,")
        ]
        tone, prefix = random.choice(tones)
        return f"{prefix} {prompt.lower()}"
    
    def _add_constraints(self, prompt: str) -> str:
        """Add constraints or requirements"""
        constraints = [
            "Be specific and detailed.",
            "Keep it clear and concise.",
            "Use simple language.",
            "Provide actionable advice.",
            "Include relevant details."
        ]
        constraint = random.choice(constraints)
        return f"{prompt} {constraint}"
    
    def _evaluate_population(
        self,
        population: List[str],
        target_examples: List[str],
        context: Optional[str] = None
    ) -> np.ndarray:
        """Evaluate all prompts in the population"""
        scores = []
        
        for prompt in population:
            # Check cache first
            cache_key = f"{prompt}_{context}_{self.objective}"
            if self.config.use_cache and cache_key in self.evaluation_cache:
                score = self.evaluation_cache[cache_key]
            else:
                score = self._evaluate_single_prompt(prompt, target_examples, context)
                if self.config.use_cache:
                    self.evaluation_cache[cache_key] = score
            
            scores.append(score)
        
        return np.array(scores)
    
    def _evaluate_single_prompt(
        self,
        prompt: str,
        target_examples: List[str],
        context: Optional[str] = None
    ) -> float:
        """Evaluate a single prompt against target examples"""
        try:
            # Generate examples with the prompt
            #import ipdb; ipdb.set_trace(context=20)
            generated_examples = self._generate_examples(
                prompt, context, self.config.num_evaluations_per_prompt
            )
            
            # Calculate different scores
            similarity_score = self._calculate_similarity_score(generated_examples, target_examples)
            perplexity_score = self._calculate_perplexity_score(generated_examples)
            diversity_score = self._calculate_diversity_score(generated_examples)
            length_score = self._calculate_length_score(generated_examples, target_examples)
            
            # Weighted combination
            final_score = (
                self.config.similarity_weight * similarity_score +
                self.config.perplexity_weight * perplexity_score +
                self.config.diversity_weight * diversity_score +
                self.config.length_weight * length_score
            )
            
            return final_score
            
        except Exception as e:
            logger.warning(f"Error evaluating prompt: {e}")
            return 0.0
    
    def _generate_examples(
        self,
        prompt: str,
        context: Optional[str] = None,
        num_examples: int = 3
    ) -> List[str]:
        """Generate examples using the given prompt"""
        examples = []
        
        # Format prompt with template if needed
        formatted_prompt = self._format_prompt(prompt, context)
        
        for _ in range(num_examples):
            try:
                # Tokenize input
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length - self.config.max_new_tokens
                ).to(self.device)
                
                # Generate using standard transformers generation
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        generation_config=self.generation_config,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        do_sample=True
                    )
                
                # Decode only the new tokens
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                examples.append(generated_text)
                
            except Exception as e:
                logger.warning(f"Error generating example: {e}")
                examples.append("[Generation failed]")
        
        return examples
    
    def _format_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """Format prompt with appropriate template"""
        if not self.config.use_instruction_template:
            return f"{context} {prompt}" if context else prompt
        
        # Use instruction template
        if context or self.config.system_message:
            system_msg = context or self.config.system_message or ""
            full_prompt = f"System: {system_msg}\n\n{self.config.instruction_template.format(instruction=prompt)}"
        else:
            full_prompt = self.config.instruction_template.format(instruction=prompt)
        
        return full_prompt
    
    def _calculate_similarity_score(
        self,
        generated_examples: List[str],
        target_examples: List[str]
    ) -> float:
        """Calculate similarity between generated and target examples"""
        if not generated_examples or not target_examples:
            return 0.0
        
        similarities = []
        for gen_example in generated_examples:
            max_sim = 0.0
            for target_example in target_examples:
                sim = self._text_similarity(gen_example, target_example)
                max_sim = max(max_sim, sim)
            similarities.append(max_sim)
        
        return np.mean(similarities)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using multiple metrics"""
        # Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        jaccard = len(words1 & words2) / len(words1 | words2)
        
        # Length ratio similarity
        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2), 1)
        
        # Combine metrics
        return 0.7 * jaccard + 0.3 * len_ratio
    
    def _calculate_perplexity_score(self, examples: List[str]) -> float:
        """Calculate perplexity-based score (lower perplexity = higher score)"""
        if not examples:
            return 0.0
        
        total_perplexity = 0.0
        valid_examples = 0
        
        for example in examples:
            if not example.strip() or example == "[Generation failed]":
                continue
                
            try:
                inputs = self.tokenizer(
                    example,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs.input_ids)
                    perplexity = torch.exp(outputs.loss).item()
                    
                if not np.isnan(perplexity) and perplexity > 0:
                    total_perplexity += perplexity
                    valid_examples += 1
                    
            except Exception as e:
                logger.debug(f"Error calculating perplexity: {e}")
                continue
        
        if valid_examples == 0:
            return 0.0
        
        avg_perplexity = total_perplexity / valid_examples
        # Convert to score (lower perplexity = higher score)
        return 1.0 / (1.0 + np.log(avg_perplexity + 1))
    
    def _calculate_diversity_score(self, examples: List[str]) -> float:
        """Calculate diversity score within generated examples"""
        if len(examples) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(examples)):
            for j in range(i + 1, len(examples)):
                sim = self._text_similarity(examples[i], examples[j])
                similarities.append(sim)
        
        # Diversity is 1 - average similarity
        return 1.0 - np.mean(similarities) if similarities else 1.0
    
    def _calculate_length_score(
        self,
        generated_examples: List[str],
        target_examples: List[str]
    ) -> float:
        """Calculate score based on length appropriateness"""
        if not generated_examples or not target_examples:
            return 0.0
        
        avg_target_length = np.mean([len(ex.split()) for ex in target_examples])
        
        length_scores = []
        for example in generated_examples:
            gen_length = len(example.split())
            # Prefer lengths similar to target examples
            ratio = min(gen_length, avg_target_length) / max(gen_length, avg_target_length, 1)
            length_scores.append(ratio)
        
        return np.mean(length_scores)
    
    def _evolve_population(self, population: List[str], scores: np.ndarray) -> List[str]:
        """Evolve population using genetic algorithm"""
        new_population = []
        
        # Elitism: keep best performers
        elite_count = max(1, self.config.population_size // 4)
        elite_indices = np.argsort(scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(population[idx])
        
        # Generate offspring through crossover and mutation
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = self._tournament_selection(population, scores)
            parent2 = self._tournament_selection(population, scores)
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                offspring = self._crossover(parent1, parent2)
            else:
                offspring = parent1 if random.random() < 0.5 else parent2
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                offspring = self._mutate(offspring)
            
            new_population.append(offspring)
        
        return new_population[:self.config.population_size]
    
    def _tournament_selection(self, population: List[str], scores: np.ndarray) -> str:
        """Tournament selection for genetic algorithm"""
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_scores = scores[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_scores)]
        return population[winner_idx]
    
    def _crossover(self, parent1: str, parent2: str) -> str:
        """Create offspring through crossover"""
        # Simple word-level crossover
        words1 = parent1.split()
        words2 = parent2.split()
        
        if not words1 or not words2:
            return parent1 if len(parent1) > len(parent2) else parent2
        
        # Take first part from parent1, second part from parent2
        crossover_point = random.randint(1, min(len(words1), len(words2)) - 1)
        offspring_words = words1[:crossover_point] + words2[crossover_point:]
        
        return " ".join(offspring_words)
    
    def _mutate(self, prompt: str) -> str:
        """Mutate a prompt"""
        mutation_strategies = [
            self._add_word_mutation,
            self._remove_word_mutation,
            self._replace_word_mutation,
            self._reorder_mutation
        ]
        
        strategy = random.choice(mutation_strategies)
        return strategy(prompt)
    
    def _add_word_mutation(self, prompt: str) -> str:
        """Add a word to the prompt"""
        additional_words = [
            "please", "carefully", "detailed", "specific", "clear",
            "comprehensive", "thorough", "precise", "accurate"
        ]
        word = random.choice(additional_words)
        words = prompt.split()
        position = random.randint(0, len(words))
        words.insert(position, word)
        return " ".join(words)
    
    def _remove_word_mutation(self, prompt: str) -> str:
        """Remove a word from the prompt"""
        words = prompt.split()
        if len(words) > 3:  # Keep minimum length
            words.pop(random.randint(0, len(words) - 1))
        return " ".join(words)
    
    def _replace_word_mutation(self, prompt: str) -> str:
        """Replace a word in the prompt"""
        replacements = {
            "write": "create",
            "explain": "describe",
            "tell": "show",
            "give": "provide",
            "make": "create",
            "good": "excellent",
            "simple": "clear"
        }
        
        words = prompt.split()
        for i, word in enumerate(words):
            if word.lower() in replacements:
                words[i] = replacements[word.lower()]
                break
        
        return " ".join(words)
    
    def _reorder_mutation(self, prompt: str) -> str:
        """Reorder parts of the prompt"""
        words = prompt.split()
        if len(words) > 4:
            # Swap two adjacent words
            pos = random.randint(0, len(words) - 2)
            words[pos], words[pos + 1] = words[pos + 1], words[pos]
        return " ".join(words)
    
    def _hill_climbing_step(self, population: List[str], scores: np.ndarray) -> List[str]:
        """Perform hill climbing optimization step"""
        best_idx = np.argmax(scores)
        best_prompt = population[best_idx]
        
        new_population = [best_prompt]  # Keep the best
        
        # Generate neighbors of the best prompt
        while len(new_population) < self.config.population_size:
            neighbor = self._mutate(best_prompt)
            new_population.append(neighbor)
        
        return new_population
    
    def _random_search_step(self, initial_prompt: str) -> List[str]:
        """Perform random search optimization step"""
        return self._create_initial_population(initial_prompt)
    
    def _hybrid_optimization_step(
        self,
        population: List[str],
        scores: np.ndarray,
        initial_prompt: str
    ) -> List[str]:
        """Combine multiple optimization strategies"""
        new_population = []
        
        # Keep best (elitism)
        best_idx = np.argmax(scores)
        new_population.append(population[best_idx])
        
        # Use genetic algorithm for half
        genetic_size = self.config.population_size // 2
        genetic_pop = self._evolve_population(population, scores)
        new_population.extend(genetic_pop[1:genetic_size])  # Skip best (already added)
        
        # Use random search for remaining
        while len(new_population) < self.config.population_size:
            random_variant = random.choice(self._create_initial_population(initial_prompt))
            new_population.append(random_variant)
        
        return new_population[:self.config.population_size]
    
    def _calculate_population_diversity(self, population: List[str]) -> float:
        """Calculate diversity within the population"""
        if len(population) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                sim = self._text_similarity(population[i], population[j])
                similarities.append(sim)
        
        return 1.0 - np.mean(similarities) if similarities else 1.0
    
    def _save_results(self, results: Dict[str, Any]):
        """Save optimization results"""
        timestamp = int(time.time())
        output_file = Path(self.config.output_dir) / f"optimization_results_{timestamp}.json"
        
        # Make results JSON serializable
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                serializable_results[k] = v.tolist()
            elif isinstance(v, torch.Tensor):
                serializable_results[k] = v.tolist()
            else:
                serializable_results[k] = v
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")


def main():
    """Command-line interface for Llama/Qwen prompt optimization"""
    parser = argparse.ArgumentParser(description="Optimize prompts for Llama/Qwen family models")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the Llama/Qwen model")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Path to tokenizer (defaults to model_path)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for inference")
    parser.add_argument("--torch_dtype", type=str, default="auto",
                       choices=["auto", "float16", "bfloat16", "float32"],
                       help="PyTorch data type for model")
    
    # Input arguments
    parser.add_argument("--prompt", type=str, required=True,
                       help="Initial prompt to optimize")
    parser.add_argument("--target_examples", type=str, nargs="+", required=True,
                       help="Target example outputs")
    parser.add_argument("--context", type=str, default=None,
                       help="Optional context or system message")
    parser.add_argument("--objective", type=str, default="similarity",
                       choices=["similarity", "diversity", "perplexity"],
                       help="Optimization objective")
    
    # Optimization arguments
    parser.add_argument("--num_iterations", type=int, default=15,
                       help="Number of optimization iterations")
    parser.add_argument("--population_size", type=int, default=8,
                       help="Population size for genetic algorithm")
    parser.add_argument("--optimization_strategy", type=str, default="genetic",
                       choices=["genetic", "random", "hill_climbing", "hybrid"],
                       help="Optimization strategy")
    parser.add_argument("--mutation_rate", type=float, default=0.3,
                       help="Mutation rate for genetic algorithm")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling parameter")
    
    # Template arguments
    parser.add_argument("--use_instruction_template", action="store_true",
                       help="Use instruction template formatting")
    parser.add_argument("--system_message", type=str, default=None,
                       help="System message for instruction template")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./optimized_prompts",
                       help="Directory to save results")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create configuration
    config = PromptOptimizationConfig(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        num_iterations=args.num_iterations,
        population_size=args.population_size,
        optimization_strategy=args.optimization_strategy,
        mutation_rate=args.mutation_rate,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        use_instruction_template=args.use_instruction_template,
        system_message=args.system_message,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # Initialize optimizer
    logger.info("Initializing Prompt Optimizer...")
    optimizer = PromptOptimizer(config)
    
    # Run optimization
    results = optimizer.optimize_prompt(
        initial_prompt=args.prompt,
        target_examples=args.target_examples,
        objective=args.objective,
        context=args.context
    )
    
    # Display results
    print("\n" + "="*70)
    print("PROMPT OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Model: {optimizer.model_name}")
    print(f"Strategy: {config.optimization_strategy}")
    print(f"Iterations: {results['total_iterations']}")
    print(f"Optimization time: {results['optimization_time']:.2f} seconds")
    print(f"\nInitial prompt:")
    print(f"  {results['initial_prompt']}")
    print(f"\nOptimized prompt:")
    print(f"  {results['optimized_prompt']}")
    print(f"\nBest score: {results['best_score']:.4f}")
    
    print(f"\nTarget examples:")
    for i, example in enumerate(results['target_examples'], 1):
        print(f"  {i}. {example}")
    
    print(f"\nGenerated examples with optimized prompt:")
    for i, example in enumerate(results['final_generated_examples'], 1):
        print(f"  {i}. {example}")
    print("="*70)


if __name__ == "__main__":
    main()
