#!/usr/bin/env python3
"""
DiffuCoder Benchmark Script
===========================

This script benchmarks the DiffuCoder-7B-cpGRPO model with different parameters
to help optimize performance for your specific use case.
"""

import torch
import time
import json
from diffucoder_example import DiffuCoderDemo
import gc


class DiffuCoderBenchmark:
    def __init__(self):
        """Initialize the benchmark with the DiffuCoder model."""
        print("🚀 Initializing DiffuCoder Benchmark")
        self.demo = DiffuCoderDemo()
        self.results = []
    
    def benchmark_parameters(self):
        """Benchmark different parameter combinations."""
        test_query = "Write a Python function to implement binary search on a sorted list."
        
        # Different parameter combinations to test
        param_combinations = [
            {"max_new_tokens": 128, "temperature": 0.4, "top_p": 0.95, "steps": 128},
            {"max_new_tokens": 256, "temperature": 0.4, "top_p": 0.95, "steps": 256},
            {"max_new_tokens": 128, "temperature": 0.2, "top_p": 0.95, "steps": 128},
            {"max_new_tokens": 128, "temperature": 0.6, "top_p": 0.95, "steps": 128},
            {"max_new_tokens": 128, "temperature": 0.4, "top_p": 0.8, "steps": 128},
            {"max_new_tokens": 128, "temperature": 0.4, "top_p": 0.95, "steps": 64},
            {"max_new_tokens": 128, "temperature": 0.4, "top_p": 0.95, "steps": 192},
        ]
        
        print(f"🎯 Testing query: {test_query}")
        print(f"📊 Running {len(param_combinations)} parameter combinations...")
        print("=" * 80)
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\n🧪 Test {i}/{len(param_combinations)}")
            print(f"Parameters: {params}")
            
            # Record start time and memory
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                start_memory = torch.cuda.memory_allocated()
            
            try:
                # Generate code with stats
                result, stats = self.demo.generate_code(test_query, return_stats=True, **params)
                
                # Record end time and memory
                end_time = time.time()
                generation_time = end_time - start_time
                
                if torch.cuda.is_available():
                    end_memory = torch.cuda.memory_allocated()
                    memory_used = (end_memory - start_memory) / 1e6  # MB
                else:
                    memory_used = 0
                
                # Use actual tokens generated from stats for more accurate measurement
                tokens_per_second = stats["tokens_per_second"]
                actual_tokens = stats["actual_tokens_generated"]
                
                # Store results
                benchmark_result = {
                    "test_id": i,
                    "parameters": params,
                    "generation_time": generation_time,
                    "tokens_per_second": tokens_per_second,
                    "actual_tokens_generated": actual_tokens,
                    "requested_tokens": params["max_new_tokens"],
                    "efficiency_percent": stats["efficiency_percent"],
                    "memory_used_mb": memory_used,
                    "result_length": len(result),
                    "success": True
                }
                
                self.results.append(benchmark_result)
                
                print(f"✅ Success!")
                print(f"⏱️  Time: {generation_time:.2f}s")
                print(f"🔥 Speed: {tokens_per_second:.1f} tokens/s")
                print(f"📊 Tokens: {actual_tokens}/{params['max_new_tokens']} ({stats['efficiency_percent']:.1f}%)")
                print(f"💾 Memory: {memory_used:.1f} MB")
                print(f"📝 Output length: {len(result)} chars")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                benchmark_result = {
                    "test_id": i,
                    "parameters": params,
                    "error": str(e),
                    "success": False
                }
                self.results.append(benchmark_result)
            
            print("-" * 80)
            
            # Clean up between tests
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def benchmark_different_queries(self):
        """Benchmark the model with different types of queries."""
        queries = [
            {
                "name": "Simple Function",
                "query": "Write a function to calculate factorial of a number.",
                "expected_complexity": "low"
            },
            {
                "name": "Data Structure",
                "query": "Create a Python class for a stack with push, pop, and peek operations.",
                "expected_complexity": "medium"
            },
            {
                "name": "Algorithm",
                "query": "Implement merge sort algorithm with detailed comments.",
                "expected_complexity": "high"
            },
            {
                "name": "Web Scraping",
                "query": "Write a function to scrape product information from an e-commerce website.",
                "expected_complexity": "high"
            },
            {
                "name": "File Processing",
                "query": "Create a function to read a CSV file and return a dictionary of column statistics.",
                "expected_complexity": "medium"
            }
        ]
        
        print("\n🎨 Benchmarking Different Query Types")
        print("=" * 80)
        
        # Use optimal parameters from previous benchmarks
        optimal_params = {
            "max_new_tokens": 200,
            "temperature": 0.4,
            "top_p": 0.95,
            "steps": 200
        }
        
        for query_info in queries:
            print(f"\n📝 Query Type: {query_info['name']}")
            print(f"Query: {query_info['query']}")
            print(f"Expected Complexity: {query_info['expected_complexity']}")
            
            start_time = time.time()
            
            try:
                result, stats = self.demo.generate_code(
                    query_info["query"], 
                    return_stats=True, 
                    **optimal_params
                )
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                print(f"✅ Generated successfully!")
                print(f"⏱️  Time: {generation_time:.2f}s")
                print(f"🔥 Speed: {stats['tokens_per_second']:.1f} tokens/s")
                print(f"📊 Tokens: {stats['actual_tokens_generated']}/{stats['requested_tokens']} ({stats['efficiency_percent']:.1f}%)")
                print(f"📄 Result preview: {result[:100]}...")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
            
            print("-" * 80)
            
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n📊 BENCHMARK SUMMARY")
        print("=" * 80)
        
        successful_tests = [r for r in self.results if r.get("success", False)]
        
        if not successful_tests:
            print("❌ No successful tests to summarize")
            return
        
        # Find best performing configuration
        best_speed = max(successful_tests, key=lambda x: x["tokens_per_second"])
        best_memory = min(successful_tests, key=lambda x: x["memory_used_mb"])
        
        print(f"🏆 Best Speed Configuration:")
        print(f"   Parameters: {best_speed['parameters']}")
        print(f"   Speed: {best_speed['tokens_per_second']:.1f} tokens/s")
        print(f"   Time: {best_speed['generation_time']:.2f}s")
        
        print(f"\n💾 Most Memory Efficient:")
        print(f"   Parameters: {best_memory['parameters']}")
        print(f"   Memory: {best_memory['memory_used_mb']:.1f} MB")
        print(f"   Speed: {best_memory['tokens_per_second']:.1f} tokens/s")
        
        # Average statistics
        avg_speed = sum(r["tokens_per_second"] for r in successful_tests) / len(successful_tests)
        avg_memory = sum(r["memory_used_mb"] for r in successful_tests) / len(successful_tests)
        avg_time = sum(r["generation_time"] for r in successful_tests) / len(successful_tests)
        
        print(f"\n📈 Average Performance:")
        print(f"   Speed: {avg_speed:.1f} tokens/s")
        print(f"   Memory: {avg_memory:.1f} MB")
        print(f"   Time: {avg_time:.2f}s")
        
        print(f"\n✅ Success Rate: {len(successful_tests)}/{len(self.results)} ({len(successful_tests)/len(self.results)*100:.1f}%)")
    
    def save_results(self, filename="benchmark_results.json"):
        """Save benchmark results to a JSON file."""
        with open(filename, "w") as f:
            json.dump({
                "timestamp": time.time(),
                "gpu_info": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
                "results": self.results
            }, f, indent=2)
        print(f"💾 Results saved to {filename}")
    
    def run_full_benchmark(self):
        """Run the complete benchmark suite."""
        print("🎯 Starting Full DiffuCoder Benchmark")
        print("=" * 80)
        
        # Run parameter benchmarks
        self.benchmark_parameters()
        
        # Run query type benchmarks
        self.benchmark_different_queries()
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
        
        print("\n🎉 Benchmark Complete!")


def main():
    """Main function to run the benchmark."""
    try:
        benchmark = DiffuCoderBenchmark()
        benchmark.run_full_benchmark()
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print("Make sure:")
        print("1. DiffuCoder model is available")
        print("2. GPU has sufficient memory")
        print("3. All dependencies are installed")
    
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main() 