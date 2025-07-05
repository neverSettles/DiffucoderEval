#!/usr/bin/env python3
"""
Large Prompt Benchmark for DiffuCoder
=====================================

This script tests DiffuCoder's performance with increasingly large prompts
to understand how tokens/second scales with context length.
"""

import torch
import time
import gc
from diffucoder_example import DiffuCoderDemo


class LargePromptBenchmark:
    def __init__(self):
        """Initialize the benchmark."""
        print("🚀 Initializing Large Prompt Benchmark")
        self.demo = DiffuCoderDemo()
        self.results = []
    
    def create_large_prompt(self, size_category="medium"):
        """Create prompts of different sizes."""
        base_requirements = """
You are tasked with creating a comprehensive Python application. Here are the detailed requirements:

1. Create a web scraping system that can handle multiple websites simultaneously
2. Implement robust error handling and retry mechanisms
3. Add logging capabilities with different log levels
4. Create a database layer using SQLAlchemy
5. Implement caching mechanisms using Redis
6. Add configuration management using environment variables
7. Create unit tests with pytest
8. Implement API endpoints using FastAPI
9. Add authentication and authorization
10. Create documentation using Sphinx
        """
        
        if size_category == "small":
            return "Write a simple function to calculate the factorial of a number."
        
        elif size_category == "medium":
            return base_requirements + """
            
Additional requirements:
- Use async/await for all I/O operations
- Implement rate limiting
- Add data validation using Pydantic
- Create Docker containerization
- Implement monitoring and health checks
"""
        
        elif size_category == "large":
            return base_requirements + """
            
Additional requirements:
- Use async/await for all I/O operations
- Implement rate limiting with sliding window algorithm
- Add comprehensive data validation using Pydantic models
- Create Docker containerization with multi-stage builds
- Implement monitoring and health checks with Prometheus metrics
- Add distributed task queue using Celery
- Implement WebSocket connections for real-time updates
- Create microservices architecture with service discovery
- Add circuit breaker pattern for external API calls
- Implement distributed caching across multiple Redis instances
- Create comprehensive API documentation with OpenAPI/Swagger
- Add performance profiling and optimization
- Implement database migrations using Alembic
- Create comprehensive integration tests
- Add security scanning and vulnerability assessment
- Implement load balancing and auto-scaling
- Create monitoring dashboards using Grafana
- Add backup and disaster recovery procedures
- Implement audit logging and compliance features
- Create performance benchmarking suite
"""
        
        elif size_category == "very_large":
            return base_requirements + """
            
Comprehensive System Requirements:

ARCHITECTURE:
- Microservices architecture with event-driven communication
- Domain-driven design with bounded contexts
- CQRS (Command Query Responsibility Segregation) pattern
- Event sourcing for audit trails
- Hexagonal architecture for clean separation
- Distributed system with multiple data centers
- Service mesh implementation using Istio
- API Gateway with rate limiting and authentication
- Message queues using Apache Kafka for event streaming
- Distributed caching using Redis Cluster

INFRASTRUCTURE:
- Kubernetes orchestration with Helm charts
- Infrastructure as Code using Terraform
- CI/CD pipelines with GitLab CI or GitHub Actions
- Container registry with automated security scanning
- Service discovery and load balancing
- Auto-scaling based on metrics and predictions
- Multi-region deployment for high availability
- Database sharding and replication
- CDN integration for static assets
- Monitoring and alerting with Prometheus and Grafana

SECURITY:
- OAuth 2.0 and OpenID Connect authentication
- Role-based access control (RBAC)
- API key management and rotation
- Encryption at rest and in transit
- Web Application Firewall (WAF)
- DDoS protection and rate limiting
- Security headers and CORS configuration
- Vulnerability scanning and penetration testing
- Compliance with GDPR, HIPAA, and SOC 2
- Audit logging and forensic capabilities

PERFORMANCE:
- Response time SLA of < 100ms for 95th percentile
- Throughput capacity of 10,000 requests per second
- Database query optimization and indexing
- Caching strategies at multiple levels
- Asynchronous processing for heavy operations
- Connection pooling and resource optimization
- Memory profiling and garbage collection tuning
- Load testing with realistic traffic patterns
- Performance monitoring and alerting
- Capacity planning and scaling strategies

DATA MANAGEMENT:
- Database design with proper normalization
- Data validation and sanitization
- Backup and restoration procedures
- Data encryption and privacy protection
- ETL pipelines for data processing
- Data warehouse for analytics
- Real-time data streaming and processing
- Data retention and archival policies
- Data quality monitoring and validation
- Master data management

MONITORING AND OBSERVABILITY:
- Distributed tracing with Jaeger or Zipkin
- Metrics collection and visualization
- Log aggregation and analysis
- Error tracking and alerting
- Performance profiling and optimization
- Health checks and uptime monitoring
- Capacity planning and resource utilization
- Cost optimization and budget tracking
- Security incident response procedures
- Disaster recovery and business continuity

QUALITY ASSURANCE:
- Unit testing with >90% code coverage
- Integration testing for all components
- End-to-end testing with automated browsers
- Performance testing and load testing
- Security testing and vulnerability assessments
- Accessibility testing and compliance
- Usability testing and user experience validation
- Code quality metrics and static analysis
- Dependency vulnerability scanning
- Continuous quality gates in CI/CD

Now, implement a comprehensive solution that addresses all these requirements with proper documentation, testing, and deployment procedures.
"""
        
        elif size_category == "extreme":
            # Create an extremely large prompt by repeating and expanding
            extreme_content = base_requirements
            
            # Add multiple detailed sections
            sections = [
                "\n\nDETAILED TECHNICAL SPECIFICATIONS:",
                "\n\nPERFORMANCE REQUIREMENTS:",
                "\n\nSECURITY SPECIFICATIONS:",
                "\n\nTESTING REQUIREMENTS:",
                "\n\nDEPLOYMENT SPECIFICATIONS:",
                "\n\nMONITORING AND ALERTING:",
                "\n\nDOCUMENTATION REQUIREMENTS:",
                "\n\nCOMPLIANCE AND GOVERNANCE:",
                "\n\nSCALABILITY REQUIREMENTS:",
                "\n\nMAINTENANCE PROCEDURES:"
            ]
            
            detailed_specs = """
- Implement distributed systems with fault tolerance
- Use database connection pooling with HikariCP
- Implement circuit breaker pattern with Hystrix
- Add distributed caching with Redis Cluster
- Use message queues for asynchronous processing
- Implement event-driven architecture with Apache Kafka
- Add comprehensive logging with structured formats
- Use distributed tracing for request correlation
- Implement health checks and readiness probes
- Add metrics collection with Prometheus
- Use configuration management with Consul
- Implement service discovery with Eureka
- Add load balancing with NGINX or HAProxy
- Use container orchestration with Kubernetes
- Implement infrastructure as code with Terraform
- Add automated testing with comprehensive coverage
- Use continuous integration and deployment
- Implement security scanning and vulnerability assessment
- Add performance monitoring and optimization
- Use backup and disaster recovery procedures
- Implement audit logging and compliance features
- Add rate limiting and throttling mechanisms
- Use data validation and sanitization
- Implement encryption for data at rest and in transit
- Add authentication and authorization mechanisms
- Use role-based access control (RBAC)
- Implement API versioning and backward compatibility
- Add comprehensive error handling and recovery
- Use graceful degradation and fallback mechanisms
- Implement resource pooling and optimization
"""
            
            for section in sections:
                extreme_content += section + detailed_specs
            
            return extreme_content
        
        else:
            return base_requirements
    
    def benchmark_prompt_sizes(self):
        """Benchmark different prompt sizes."""
        print("🎯 Benchmarking Different Prompt Sizes")
        print("=" * 80)
        
        sizes = ["small", "medium", "large", "very_large", "extreme"]
        
        for size in sizes:
            print(f"\n📏 Testing {size.upper()} prompt...")
            
            # Create the prompt
            prompt = self.create_large_prompt(size)
            prompt_tokens = len(self.demo.tokenizer.encode(prompt))
            
            print(f"📊 Prompt length: {len(prompt)} characters")
            print(f"🔢 Prompt tokens: {prompt_tokens}")
            
            # Test with different generation lengths
            test_configs = [
                {"max_new_tokens": 128, "desc": "Short generation"},
                {"max_new_tokens": 256, "desc": "Medium generation"},
                {"max_new_tokens": 512, "desc": "Long generation"}
            ]
            
            for config in test_configs:
                print(f"\n  🧪 {config['desc']} ({config['max_new_tokens']} tokens)")
                
                try:
                    start_time = time.time()
                    
                    # Generate with stats
                    result, stats = self.demo.generate_code(
                        prompt,
                        max_new_tokens=config["max_new_tokens"],
                        temperature=0.4,
                        return_stats=True
                    )
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    # Calculate context efficiency
                    context_ratio = prompt_tokens / (prompt_tokens + stats["actual_tokens_generated"])
                    
                    # Store results
                    result_data = {
                        "prompt_size": size,
                        "prompt_tokens": prompt_tokens,
                        "prompt_chars": len(prompt),
                        "generation_config": config["desc"],
                        "tokens_per_second": stats["tokens_per_second"],
                        "actual_tokens": stats["actual_tokens_generated"],
                        "total_time": total_time,
                        "efficiency": stats["efficiency_percent"],
                        "context_ratio": context_ratio
                    }
                    
                    self.results.append(result_data)
                    
                    print(f"    ✅ Success!")
                    print(f"    🔥 Speed: {stats['tokens_per_second']:.1f} tokens/s")
                    print(f"    📊 Efficiency: {stats['efficiency_percent']:.1f}%")
                    print(f"    ⏱️  Time: {total_time:.2f}s")
                    print(f"    📈 Context ratio: {context_ratio:.2%}")
                    
                except Exception as e:
                    print(f"    ❌ Failed: {e}")
                    # Store failure data
                    self.results.append({
                        "prompt_size": size,
                        "prompt_tokens": prompt_tokens,
                        "prompt_chars": len(prompt),
                        "generation_config": config["desc"],
                        "error": str(e),
                        "success": False
                    })
                
                # Clean up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            print("-" * 80)
    
    def test_extreme_context(self):
        """Test with extremely large context to find limits."""
        print("\n🚀 Testing Extreme Context Limits")
        print("=" * 80)
        
        # Start with a reasonable size and progressively increase
        multipliers = [1, 2, 4, 8, 16]
        base_prompt = self.create_large_prompt("very_large")
        
        for multiplier in multipliers:
            # Create progressively larger prompts
            large_prompt = base_prompt * multiplier
            prompt_tokens = len(self.demo.tokenizer.encode(large_prompt))
            
            print(f"\n📏 Testing {multiplier}x size:")
            print(f"   📊 Characters: {len(large_prompt):,}")
            print(f"   🔢 Tokens: {prompt_tokens:,}")
            
            # Check GPU memory before attempting
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated() / 1e9
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"   💾 GPU Memory: {memory_before:.1f}/{memory_total:.1f} GB")
            
            try:
                start_time = time.time()
                
                result, stats = self.demo.generate_code(
                    large_prompt,
                    max_new_tokens=128,  # Keep generation short for large contexts
                    temperature=0.4,
                    return_stats=True
                )
                
                end_time = time.time()
                
                print(f"   ✅ Success! Speed: {stats['tokens_per_second']:.1f} tokens/s")
                print(f"   ⏱️  Time: {end_time - start_time:.2f}s")
                
                # Store successful result
                self.results.append({
                    "test_type": "extreme_context",
                    "multiplier": multiplier,
                    "prompt_tokens": prompt_tokens,
                    "tokens_per_second": stats["tokens_per_second"],
                    "success": True
                })
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                print(f"   🔥 Maximum context reached at {multiplier-1}x size")
                
                # Store failure point
                self.results.append({
                    "test_type": "extreme_context",
                    "multiplier": multiplier,
                    "prompt_tokens": prompt_tokens,
                    "error": str(e),
                    "success": False
                })
                
                break  # Stop at first failure
            
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def analyze_results(self):
        """Analyze and display results."""
        print("\n📊 LARGE PROMPT ANALYSIS")
        print("=" * 80)
        
        # Filter successful results
        successful_results = [r for r in self.results if r.get("success", True)]
        
        if not successful_results:
            print("❌ No successful results to analyze")
            return
        
        # Group by prompt size
        size_groups = {}
        for result in successful_results:
            if "prompt_size" in result:
                size = result["prompt_size"]
                if size not in size_groups:
                    size_groups[size] = []
                size_groups[size].append(result)
        
        print("🔍 Performance by Prompt Size:")
        for size, results in size_groups.items():
            avg_speed = sum(r["tokens_per_second"] for r in results) / len(results)
            avg_tokens = sum(r["prompt_tokens"] for r in results) / len(results)
            print(f"   📏 {size.upper()}: {avg_speed:.1f} tokens/s (avg {avg_tokens:.0f} prompt tokens)")
        
        # Find performance correlation with context length
        prompt_sizes = [r["prompt_tokens"] for r in successful_results if "prompt_tokens" in r]
        speeds = [r["tokens_per_second"] for r in successful_results if "tokens_per_second" in r]
        
        if prompt_sizes and speeds:
            print(f"\n📈 Context Length vs Speed Analysis:")
            print(f"   📊 Smallest prompt: {min(prompt_sizes)} tokens")
            print(f"   📊 Largest prompt: {max(prompt_sizes)} tokens")
            print(f"   🔥 Speed range: {min(speeds):.1f} - {max(speeds):.1f} tokens/s")
            
            # Simple correlation analysis
            if len(prompt_sizes) > 1:
                import statistics
                speed_std = statistics.stdev(speeds)
                print(f"   📊 Speed variation: ±{speed_std:.1f} tokens/s")
        
        # Show extreme context results
        extreme_results = [r for r in self.results if r.get("test_type") == "extreme_context"]
        if extreme_results:
            successful_extreme = [r for r in extreme_results if r.get("success", True)]
            if successful_extreme:
                max_multiplier = max(r["multiplier"] for r in successful_extreme)
                max_tokens = max(r["prompt_tokens"] for r in successful_extreme)
                print(f"\n🚀 Maximum Context Achieved:")
                print(f"   📏 Multiplier: {max_multiplier}x")
                print(f"   🔢 Tokens: {max_tokens:,}")
        
        print("=" * 80)
    
    def run_comprehensive_test(self):
        """Run the complete large prompt benchmark."""
        print("🎯 Starting Comprehensive Large Prompt Benchmark")
        print("=" * 80)
        
        # Test different prompt sizes
        self.benchmark_prompt_sizes()
        
        # Test extreme context limits
        self.test_extreme_context()
        
        # Analyze results
        self.analyze_results()
        
        print("\n🎉 Large Prompt Benchmark Complete!")


def main():
    """Main function."""
    try:
        benchmark = LargePromptBenchmark()
        benchmark.run_comprehensive_test()
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print("Make sure you have sufficient GPU memory for large prompts")
    
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main() 