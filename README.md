# prompt_optimizer

A prompt optimization tool for Transformer models using basic approaches with iterative refinement. Automatically optimizes prompts by evolving them through multiple generations to achieve better performance on specific tasks.

## Optimization Strategies

The tool implements four different optimization strategies:

### 1. **Genetic Algorithm** (`optimization_strategy: "genetic"`)
- Population-based evolution with elitism, tournament selection, crossover, and mutation
- Uses tournament selection to choose parents and combines them through crossover
- Applies various mutation operations to introduce diversity

### 2. **Hill Climbing** (`optimization_strategy: "hill_climbing"`)
- Local search starting from the best candidate in the current population
- Generates neighbors through mutation of the best-performing prompt
- Focuses on exploiting local optima

### 3. **Random Search** (`optimization_strategy: "random"`)
- Generates random variations of the initial prompt each iteration
- Provides good exploration of the search space
- Simple but effective baseline approach

### 4. **Hybrid Strategy** (`optimization_strategy: "hybrid"`)
- Combines genetic algorithm (50% of population) with random search
- Includes elitism to preserve best candidates
- Balances exploration and exploitation

## Usage

```bash
python example_usage.py
```
