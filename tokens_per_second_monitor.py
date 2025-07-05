#!/usr/bin/env python3
"""
Tokens/Second Monitor for DiffuCoder
===================================

A simple script to monitor tokens/second performance with real-time updates.
Perfect for optimizing parameters and monitoring performance during use.
"""

import time
import torch
from diffucoder_example import DiffuCoderDemo


class TokensPerSecondMonitor:
    def __init__(self):
        """Initialize the monitor with DiffuCoder."""
        print("🚀 Initializing Tokens/Second Monitor")
        self.demo = DiffuCoderDemo()
        self.performance_history = []
    
    def quick_test(self, max_new_tokens=128, temperature=0.4, top_p=0.95):
        """Run a quick test to measure tokens/second."""
        test_query = "Write a simple Python function to add two numbers."
        
        print(f"\n🎯 Quick Test (max_tokens={max_new_tokens}, temp={temperature}, top_p={top_p})")
        print("-" * 60)
        
        try:
            result, stats = self.demo.generate_code(
                test_query,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                return_stats=True
            )
            
            # Store performance data
            self.performance_history.append({
                "timestamp": time.time(),
                "tokens_per_second": stats["tokens_per_second"],
                "actual_tokens": stats["actual_tokens_generated"],
                "efficiency": stats["efficiency_percent"],
                "generation_time": stats["generation_time"]
            })
            
            print(f"🔥 TOKENS/SECOND: {stats['tokens_per_second']:.1f}")
            print(f"📊 Efficiency: {stats['efficiency_percent']:.1f}%")
            print(f"⏱️  Time: {stats['generation_time']:.2f}s")
            
            return stats["tokens_per_second"]
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return 0
    
    def parameter_sweep(self):
        """Test different parameters to find optimal tokens/second."""
        print("\n🔍 Parameter Sweep for Optimal Tokens/Second")
        print("=" * 60)
        
        # Test different configurations
        configs = [
            {"max_new_tokens": 64, "temperature": 0.2, "top_p": 0.9},
            {"max_new_tokens": 128, "temperature": 0.2, "top_p": 0.9},
            {"max_new_tokens": 128, "temperature": 0.4, "top_p": 0.95},
            {"max_new_tokens": 256, "temperature": 0.4, "top_p": 0.95},
            {"max_new_tokens": 128, "temperature": 0.6, "top_p": 0.95},
            {"max_new_tokens": 128, "temperature": 0.4, "top_p": 0.8},
        ]
        
        best_speed = 0
        best_config = None
        
        for i, config in enumerate(configs, 1):
            print(f"\n🧪 Config {i}/{len(configs)}: {config}")
            speed = self.quick_test(**config)
            
            if speed > best_speed:
                best_speed = speed
                best_config = config
                print(f"🏆 New best speed!")
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"\n🎉 OPTIMAL CONFIGURATION:")
        print(f"   Config: {best_config}")
        print(f"   Speed: {best_speed:.1f} tokens/second")
        
        return best_config, best_speed
    
    def continuous_monitoring(self, interval=30):
        """Run continuous monitoring of tokens/second."""
        print(f"\n📊 Continuous Monitoring (every {interval}s)")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        test_query = "Write a function to calculate the factorial of a number."
        
        try:
            while True:
                timestamp = time.strftime("%H:%M:%S")
                print(f"\n⏰ {timestamp}")
                
                result, stats = self.demo.generate_code(
                    test_query,
                    max_new_tokens=128,
                    temperature=0.4,
                    return_stats=True
                )
                
                print(f"🔥 {stats['tokens_per_second']:.1f} tokens/s | "
                      f"📊 {stats['efficiency_percent']:.1f}% efficient | "
                      f"⏱️ {stats['generation_time']:.2f}s")
                
                # Store data
                self.performance_history.append({
                    "timestamp": time.time(),
                    "tokens_per_second": stats["tokens_per_second"],
                    "actual_tokens": stats["actual_tokens_generated"],
                    "efficiency": stats["efficiency_percent"],
                    "generation_time": stats["generation_time"]
                })
                
                # Clean up and wait
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
    
    def show_performance_summary(self):
        """Show a summary of performance history."""
        if not self.performance_history:
            print("❌ No performance data available")
            return
        
        print("\n📈 PERFORMANCE SUMMARY")
        print("=" * 60)
        
        speeds = [p["tokens_per_second"] for p in self.performance_history]
        efficiencies = [p["efficiency"] for p in self.performance_history]
        times = [p["generation_time"] for p in self.performance_history]
        
        print(f"📊 Tests run: {len(self.performance_history)}")
        print(f"🔥 Average speed: {sum(speeds)/len(speeds):.1f} tokens/s")
        print(f"🏆 Best speed: {max(speeds):.1f} tokens/s")
        print(f"🐌 Worst speed: {min(speeds):.1f} tokens/s")
        print(f"📊 Average efficiency: {sum(efficiencies)/len(efficiencies):.1f}%")
        print(f"⏱️  Average time: {sum(times)/len(times):.2f}s")
        
        # Show recent performance
        if len(self.performance_history) >= 5:
            recent = self.performance_history[-5:]
            recent_avg = sum(p["tokens_per_second"] for p in recent) / len(recent)
            print(f"📈 Recent average (last 5): {recent_avg:.1f} tokens/s")
    
    def interactive_menu(self):
        """Interactive menu for tokens/second monitoring."""
        while True:
            print("\n🎮 TOKENS/SECOND MONITOR")
            print("=" * 40)
            print("1. Quick Test")
            print("2. Parameter Sweep")
            print("3. Continuous Monitoring")
            print("4. Performance Summary")
            print("5. Exit")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                self.quick_test()
            elif choice == "2":
                self.parameter_sweep()
            elif choice == "3":
                interval = input("Monitoring interval (seconds, default 30): ").strip()
                interval = int(interval) if interval.isdigit() else 30
                self.continuous_monitoring(interval)
            elif choice == "4":
                self.show_performance_summary()
            elif choice == "5":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option")


def main():
    """Main function."""
    try:
        monitor = TokensPerSecondMonitor()
        monitor.interactive_menu()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure DiffuCoder is properly installed and GPU is available")
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 