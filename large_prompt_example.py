#!/usr/bin/env python3
"""
Simple Large Prompt Example
===========================

A simple example showing how to test your own large prompts with DiffuCoder.
"""

import torch
from diffucoder_example import DiffuCoderDemo


def test_custom_large_prompt():
    """Test a custom large prompt with performance monitoring."""
    
    # Your custom large prompt
    large_prompt = """
    Create a comprehensive e-commerce platform with the following features:
    
    FRONTEND REQUIREMENTS:
    - React.js with TypeScript for the main application
    - Redux for state management
    - Material-UI for component library
    - Responsive design for mobile and desktop
    - Progressive Web App (PWA) capabilities
    - Shopping cart functionality
    - User authentication and profile management
    - Product search and filtering
    - Payment gateway integration
    - Order tracking and history
    - Wishlist and favorites
    - Product reviews and ratings
    - Social media integration
    - Live chat support
    - Multi-language support
    - Dark/light theme toggle
    
    BACKEND REQUIREMENTS:
    - Node.js with Express.js framework
    - MongoDB for database with Mongoose ODM
    - JWT for authentication
    - RESTful API design
    - GraphQL for advanced queries
    - File upload handling for product images
    - Email service integration
    - SMS notification service
    - Payment processing with Stripe
    - Inventory management system
    - Order management system
    - User management and roles
    - Analytics and reporting
    - Caching with Redis
    - Rate limiting and security
    - API documentation with Swagger
    - Unit and integration testing
    - Docker containerization
    - CI/CD pipeline setup
    
    ADDITIONAL FEATURES:
    - Admin dashboard for managing products, orders, and users
    - Vendor/seller portal for multi-vendor marketplace
    - Real-time notifications
    - Advanced search with Elasticsearch
    - Recommendation engine
    - A/B testing framework
    - Performance monitoring
    - Error tracking and logging
    - Backup and disaster recovery
    - GDPR compliance
    - Security scanning and penetration testing
    - Load testing and optimization
    - SEO optimization
    - Social media marketing integration
    - Email marketing campaigns
    - Affiliate program management
    - Loyalty program and rewards
    - Customer support ticket system
    - Knowledge base and FAQ
    - Return and refund management
    
    Please provide a complete implementation with detailed code examples, 
    database schemas, API endpoints, and deployment instructions.
    """
    
    print("🚀 Testing Custom Large Prompt")
    print("=" * 60)
    
    # Initialize DiffuCoder
    demo = DiffuCoderDemo()
    
    # Count tokens in the prompt
    prompt_tokens = len(demo.tokenizer.encode(large_prompt))
    print(f"📊 Prompt length: {len(large_prompt)} characters")
    print(f"🔢 Prompt tokens: {prompt_tokens}")
    
    # Test different generation lengths
    test_configs = [
        {"max_new_tokens": 256, "desc": "Medium response"},
        {"max_new_tokens": 512, "desc": "Long response"},
        {"max_new_tokens": 1024, "desc": "Very long response"}
    ]
    
    for config in test_configs:
        print(f"\n🧪 Testing {config['desc']} ({config['max_new_tokens']} tokens)")
        
        try:
            # Generate with detailed stats
            result, stats = demo.generate_code(
                large_prompt,
                max_new_tokens=config["max_new_tokens"],
                temperature=0.4,
                return_stats=True
            )
            
            # Calculate context ratio
            context_ratio = prompt_tokens / (prompt_tokens + stats["actual_tokens_generated"])
            
            print(f"✅ Success!")
            print(f"🔥 Speed: {stats['tokens_per_second']:.1f} tokens/s")
            print(f"📊 Efficiency: {stats['efficiency_percent']:.1f}%")
            print(f"⏱️  Time: {stats['generation_time']:.2f}s")
            print(f"📈 Context ratio: {context_ratio:.2%}")
            print(f"📄 Response preview:")
            print(f"   {result[:200]}...")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("-" * 60)


def test_prompt_size_comparison():
    """Compare performance across different prompt sizes."""
    
    print("\n🔍 Prompt Size Comparison")
    print("=" * 60)
    
    demo = DiffuCoderDemo()
    
    # Different sized prompts
    prompts = {
        "Small": "Write a function to sort a list of numbers.",
        "Medium": """
        Create a web application with the following features:
        - User authentication
        - Database integration
        - RESTful API
        - Frontend with React
        - Unit testing
        - Docker deployment
        """,
        "Large": """
        Build a comprehensive social media platform with:
        - User registration and authentication
        - Profile management with photo uploads
        - Friend/follower system
        - Post creation, editing, and deletion
        - Comments and likes functionality
        - Real-time messaging system
        - Notification system
        - News feed algorithm
        - Search functionality
        - Privacy settings
        - Content moderation
        - Analytics dashboard
        - Mobile responsive design
        - API rate limiting
        - Database optimization
        - Caching strategies
        - Security measures
        - Performance monitoring
        - Backup and recovery
        - Scalability planning
        """
    }
    
    results = []
    
    for size, prompt in prompts.items():
        print(f"\n📏 Testing {size} prompt:")
        
        prompt_tokens = len(demo.tokenizer.encode(prompt))
        print(f"   🔢 Tokens: {prompt_tokens}")
        
        try:
            result, stats = demo.generate_code(
                prompt,
                max_new_tokens=200,
                temperature=0.4,
                return_stats=True
            )
            
            results.append({
                "size": size,
                "prompt_tokens": prompt_tokens,
                "speed": stats["tokens_per_second"],
                "efficiency": stats["efficiency_percent"],
                "time": stats["generation_time"]
            })
            
            print(f"   🔥 Speed: {stats['tokens_per_second']:.1f} tokens/s")
            print(f"   📊 Efficiency: {stats['efficiency_percent']:.1f}%")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Show comparison summary
    if results:
        print(f"\n🏆 Performance Summary:")
        print(f"   Size        | Tokens | Speed    | Efficiency")
        print(f"   ------------|--------|----------|----------")
        for r in results:
            print(f"   {r['size']:<11} | {r['prompt_tokens']:<6} | {r['speed']:<8.1f} | {r['efficiency']:<8.1f}%")
        
        # Find best performer
        best = max(results, key=lambda x: x["speed"])
        print(f"\n🥇 Best performer: {best['size']} ({best['speed']:.1f} tokens/s)")


def main():
    """Main function."""
    try:
        print("🎯 Large Prompt Testing Examples")
        print("=" * 60)
        
        # Test custom large prompt
        test_custom_large_prompt()
        
        # Compare different prompt sizes
        test_prompt_size_comparison()
        
        print("\n🎉 Large prompt testing complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure DiffuCoder is properly installed and GPU is available")
    
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 