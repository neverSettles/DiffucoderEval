#!/usr/bin/env python3
"""
DiffuCoder-7B-cpGRPO Example Script
===================================

This script demonstrates how to use Apple's DiffuCoder-7B-cpGRPO model for code generation.
DiffuCoder is a diffusion-based large language model specifically designed for code generation.

The model uses a diffusion process instead of traditional autoregressive generation,
which can provide better code quality and reduce reliance on AR bias during decoding.

Requirements:
- A100 GPU or similar high-end GPU
- CUDA-compatible PyTorch installation
- Sufficient VRAM (model is ~7B parameters)

Usage:
    python diffucoder_example.py
"""

import torch
from transformers import AutoModel, AutoTokenizer
import time
import gc


class DiffuCoderDemo:
    def __init__(self, model_path="apple/DiffuCoder-7B-cpGRPO", device="cuda"):
        """
        Initialize the DiffuCoder model for code generation.
        
        Args:
            model_path: HuggingFace model path
            device: Device to run the model on
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        print(f"🚀 Initializing DiffuCoder model: {model_path}")
        print(f"📍 Device: {device}")
        
        # Check GPU availability and memory
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available, but device='cuda' specified")
            print(f"🔥 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self._load_model()
    
    def _load_model(self):
        """Load the DiffuCoder model and tokenizer."""
        print("⏳ Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        print("⏳ Loading model...")
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency on A100
            trust_remote_code=True,
            device_map="auto"  # Automatically handle device placement
        )
        
        self.model = self.model.to(self.device).eval()
        print("✅ Model loaded successfully!")
        
        # Print model info
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 Model parameters: {num_params:,} ({num_params/1e9:.1f}B)")
    
    def format_prompt(self, query, system_message="You are a helpful assistant."):
        """
        Format the prompt using the chat template expected by DiffuCoder.
        
        Args:
            query: The user's code generation request
            system_message: System message for the model
            
        Returns:
            Formatted prompt string
        """
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{query.strip()}
<|im_end|>
<|im_start|>assistant
"""
    
    def generate_code(self, query, max_new_tokens=256, steps=None, temperature=0.4, 
                     top_p=0.95, alg="entropy", alg_temp=0.0, token_per_step=1, 
                     return_stats=False):
        """
        Generate code using the diffusion process.
        
        Args:
            query: The code generation request
            max_new_tokens: Maximum number of tokens to generate
            steps: Number of diffusion steps (defaults to max_new_tokens//token_per_step)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            alg: Algorithm for diffusion ('entropy' or others)
            alg_temp: Algorithm temperature
            token_per_step: Tokens generated per diffusion step
            return_stats: If True, return tuple of (result, stats_dict)
            
        Returns:
            Generated code as a string, or tuple of (result, stats) if return_stats=True
        """
        # Format the prompt
        prompt = self.format_prompt(query)
        
        # Set default steps if not provided
        if steps is None:
            steps = max_new_tokens // token_per_step
        
        print(f"🎯 Query: {query}")
        print(f"⚙️  Parameters: max_tokens={max_new_tokens}, steps={steps}, temp={temperature}")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device=self.device)
        attention_mask = inputs.attention_mask.to(device=self.device)
        
        # Generate with diffusion
        start_time = time.time()
        
        with torch.no_grad():
            output = self.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                output_history=True,
                return_dict_in_generate=True,
                steps=steps,
                temperature=temperature,
                top_p=top_p,
                alg=alg,
                alg_temp=alg_temp,
            )
        
        generation_time = time.time() - start_time
        
        # Decode the generated text
        generations = [
            self.tokenizer.decode(g[len(p):].tolist())
            for p, g in zip(input_ids, output.sequences)
        ]
        
        # Calculate actual tokens generated (more accurate than max_new_tokens)
        actual_tokens_generated = len(output.sequences[0]) - len(input_ids[0])
        
        # Clean up the output (remove padding tokens)
        result = generations[0].split('<|dlm_pad|>')[0]
        
        # Calculate performance metrics
        tokens_per_second = actual_tokens_generated / generation_time
        
        print(f"⏱️  Generation time: {generation_time:.2f}s")
        print(f"📊 Tokens generated: {actual_tokens_generated} (requested: {max_new_tokens})")
        print(f"🔥 Tokens/second: {tokens_per_second:.1f}")
        print(f"🎯 Efficiency: {(actual_tokens_generated/max_new_tokens)*100:.1f}% of requested tokens")
        print("-" * 80)
        
        if return_stats:
            stats = {
                "generation_time": generation_time,
                "actual_tokens_generated": actual_tokens_generated,
                "requested_tokens": max_new_tokens,
                "tokens_per_second": tokens_per_second,
                "efficiency_percent": (actual_tokens_generated/max_new_tokens)*100,
                "steps": steps,
                "temperature": temperature,
                "top_p": top_p
            }
            return result.strip(), stats
        
        return result.strip()
    
    def run_examples(self):
        """Run a series of example code generation tasks."""
        examples = [
            {
                "query": "Write a function to find the shared elements from the given two lists.",
                "max_tokens": 150,
                "type": "Simple Function"
            },
            {
                "query": "Create a Python class for a binary search tree with insert, search, and delete methods.",
                "max_tokens": 300,
                "type": "Data Structure"
            },
            {
                "query": "Write a function to implement quicksort algorithm with detailed comments.",
                "max_tokens": 250,
                "type": "Algorithm"
            },
            {
                "query": "Create a decorator that measures the execution time of a function.",
                "max_tokens": 200,
                "type": "Decorator"
            },
            {
                "query": "Write a function to parse a JSON file and handle potential errors gracefully.",
                "max_tokens": 200,
                "type": "File Processing"
            }
        ]
        
        print("🎬 Running DiffuCoder examples...")
        print("=" * 80)
        
        performance_results = []
        
        for i, example in enumerate(examples, 1):
            print(f"\n📝 Example {i}/{len(examples)} - {example['type']}")
            result, stats = self.generate_code(
                example["query"], 
                max_new_tokens=example["max_tokens"],
                return_stats=True
            )
            print(f"💻 Generated Code:\n{result}")
            print(f"\n🏆 Performance Stats:")
            print(f"   🔥 Speed: {stats['tokens_per_second']:.1f} tokens/second")
            print(f"   📊 Efficiency: {stats['efficiency_percent']:.1f}%")
            print(f"   ⏱️  Time: {stats['generation_time']:.2f}s")
            print(f"   📈 Tokens: {stats['actual_tokens_generated']}/{stats['requested_tokens']}")
            print("\n" + "=" * 80)
            
            # Store performance data for summary
            performance_results.append({
                "type": example["type"],
                "speed": stats["tokens_per_second"],
                "efficiency": stats["efficiency_percent"],
                "time": stats["generation_time"],
                "tokens": stats["actual_tokens_generated"]
            })
            
            # Clean up GPU memory between examples
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Show performance summary
        self._show_performance_summary(performance_results)
    
    def _show_performance_summary(self, performance_results):
        """Show a summary of performance across different example types."""
        print("\n🏆 PERFORMANCE SUMMARY ACROSS EXAMPLES")
        print("=" * 80)
        
        # Sort by speed (descending)
        sorted_results = sorted(performance_results, key=lambda x: x["speed"], reverse=True)
        
        print("🚀 Speed Ranking (tokens/second):")
        for i, result in enumerate(sorted_results, 1):
            print(f"   {i}. {result['type']}: {result['speed']:.1f} tokens/s")
        
        print(f"\n📊 Average Performance:")
        avg_speed = sum(r["speed"] for r in performance_results) / len(performance_results)
        avg_efficiency = sum(r["efficiency"] for r in performance_results) / len(performance_results)
        avg_time = sum(r["time"] for r in performance_results) / len(performance_results)
        total_tokens = sum(r["tokens"] for r in performance_results)
        
        print(f"   🔥 Average Speed: {avg_speed:.1f} tokens/s")
        print(f"   📈 Average Efficiency: {avg_efficiency:.1f}%")
        print(f"   ⏱️  Average Time: {avg_time:.2f}s")
        print(f"   📊 Total Tokens Generated: {total_tokens}")
        
        # Find best and worst performers
        best = max(performance_results, key=lambda x: x["speed"])
        worst = min(performance_results, key=lambda x: x["speed"])
        
        print(f"\n🏆 Best Performer: {best['type']} ({best['speed']:.1f} tokens/s)")
        print(f"🐌 Slowest: {worst['type']} ({worst['speed']:.1f} tokens/s)")
        
        # Speed difference
        speed_diff = ((best["speed"] - worst["speed"]) / worst["speed"]) * 100
        print(f"📈 Speed Difference: {speed_diff:.1f}% faster")
        
        # Fun insights based on performance
        if best["type"] == "Simple Function":
            print(f"\n🎯 Insight: DiffuCoder excels at simple functions!")
        elif best["type"] == "Data Structure":
            print(f"\n🎯 Insight: DiffuCoder rocks at data structures!")
        elif best["type"] == "Algorithm":
            print(f"\n🎯 Insight: DiffuCoder is an algorithm speed demon!")
        elif best["type"] == "Decorator":
            print(f"\n🎯 Insight: DiffuCoder loves Python decorators!")
        elif best["type"] == "File Processing":
            print(f"\n🎯 Insight: DiffuCoder handles file operations like a pro!")
        
        print("=" * 80)
    
    def interactive_mode(self):
        """Run the model in interactive mode."""
        print("\n🎮 Interactive Mode - Enter your code generation requests!")
        print("Type 'quit' to exit, 'help' for options")
        
        while True:
            try:
                query = input("\n💬 Enter your request: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif query.lower() == 'help':
                    print("""
Available commands:
- Enter any code generation request
- 'quit' or 'exit' to stop
- 'help' to see this message
                    """)
                    continue
                elif not query:
                    continue
                
                # Generate code
                result = self.generate_code(query)
                print(f"💻 Generated Code:\n{result}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def main():
    """Main function to run the DiffuCoder demo."""
    print("🎯 DiffuCoder-7B-cpGRPO Demo")
    print("=" * 50)
    
    try:
        # Initialize the model
        demo = DiffuCoderDemo()
        
        # Run examples
        demo.run_examples()
        
        # Optional: Run interactive mode
        print("\n" + "=" * 50)
        choice = input("Would you like to try interactive mode? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            demo.interactive_mode()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have:")
        print("1. CUDA-compatible PyTorch installed")
        print("2. Sufficient GPU memory (A100 recommended)")
        print("3. Internet connection for model download")
        
    finally:
        # Cleanup
        if 'demo' in locals():
            demo.cleanup()


if __name__ == "__main__":
    main() 