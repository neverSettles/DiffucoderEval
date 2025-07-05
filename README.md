# DiffuCoder-7B-cpGRPO Example

This repository contains example code for using Apple's **DiffuCoder-7B-cpGRPO** model, a diffusion-based large language model specifically designed for code generation.

## About DiffuCoder

DiffuCoder is a novel approach to code generation that uses diffusion models instead of traditional autoregressive generation. Key features:

- **Diffusion-based generation**: Uses a diffusion process for better code quality
- **Reduced AR bias**: Less reliance on autoregressive bias during decoding
- **Code-specific training**: Specialized for code generation tasks
- **7B parameters**: Efficient yet powerful model size

### Model Details

- **Model**: `apple/DiffuCoder-7B-cpGRPO`
- **Paper**: [DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation](https://arxiv.org/abs/2506.20639)
- **Base Model**: Qwen2.5-7B-Coder
- **Training**: Reinforcement learning via Coupled-GRPO

## Requirements

### Hardware
- **GPU**: A100 or similar high-end GPU recommended
- **VRAM**: At least 16GB (for bfloat16 precision)
- **RAM**: 32GB+ recommended

### Software
- Python 3.8+
- CUDA-compatible PyTorch
- Internet connection (for model download)

## Quick Start

1. **Clone and setup**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Or use the setup script
   python setup_and_run.py
   ```

2. **Run the example**:
   ```bash
   python diffucoder_example.py
   ```

3. **Monitor performance**:
   ```bash
   # Quick tokens/second check
   python tokens_per_second_monitor.py
   
   # Full benchmark suite
   python benchmark_diffucoder.py
   
   # Test with very large prompts
   python large_prompt_benchmark.py
   ```

## Files Overview

### `diffucoder_example.py`
The main example script that demonstrates DiffuCoder usage:
- **DiffuCoderDemo class**: Handles model loading and generation
- **Multiple examples**: Various code generation tasks
- **Interactive mode**: Real-time code generation
- **GPU optimization**: Efficient memory usage for A100
- **Detailed tokens/second tracking**: Accurate performance metrics

### `tokens_per_second_monitor.py`
Dedicated tokens/second performance monitoring:
- **Quick tests**: Fast performance checks
- **Parameter sweeps**: Find optimal settings for speed
- **Continuous monitoring**: Real-time performance tracking
- **Performance history**: Track performance over time
- **Interactive menu**: Easy-to-use interface

### `benchmark_diffucoder.py`
Comprehensive benchmarking suite:
- **Parameter combinations**: Test different settings
- **Query types**: Benchmark various code generation tasks
- **Detailed metrics**: Tokens/second, memory usage, efficiency
- **Performance comparison**: Find optimal configurations
- **JSON export**: Save results for analysis

### `large_prompt_benchmark.py`
Large prompt performance testing:
- **Prompt size scaling**: Test small to extremely large prompts
- **Context limit testing**: Find maximum context length
- **Performance correlation**: Analyze speed vs context size
- **Memory monitoring**: Track GPU usage with large contexts
- **Scaling analysis**: Understand how performance changes with prompt size

### `requirements.txt`
Python dependencies needed to run the example:
- `torch>=2.0.0`
- `transformers>=4.36.0`
- `accelerate>=0.20.0`
- `safetensors>=0.3.0`
- `sentencepiece>=0.1.99`
- `protobuf>=3.20.0`

### `setup_and_run.py`
Automated setup script that:
- Checks GPU availability
- Installs dependencies
- Runs the example

## Usage Examples

### Basic Code Generation
```python
from diffucoder_example import DiffuCoderDemo

demo = DiffuCoderDemo()
result = demo.generate_code("Write a function to reverse a string")
print(result)
```

### Custom Parameters
```python
result = demo.generate_code(
    "Create a sorting algorithm",
    max_new_tokens=300,
    temperature=0.6,
    top_p=0.9
)
```

### Batch Processing
```python
queries = [
    "Write a binary search function",
    "Create a linked list class",
    "Implement a hash table"
]

for query in queries:
    result = demo.generate_code(query)
    print(f"Query: {query}")
    print(f"Result: {result}")
    print("-" * 50)
```

## Key Features

### Diffusion Generation
Unlike traditional autoregressive models, DiffuCoder uses a diffusion process:
- **Steps**: Number of diffusion steps (default: max_tokens // token_per_step)
- **Algorithm**: Diffusion algorithm ('entropy' by default)
- **Token per step**: Tokens generated per diffusion step

### Optimizations for A100
- **bfloat16 precision**: Reduces memory usage
- **Device mapping**: Automatic GPU placement
- **Memory cleanup**: Efficient VRAM management
- **Batch processing**: Optimized for multiple generations

### Chat Template
DiffuCoder uses a specific chat template format:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant
```

## Performance Tips

### Tokens/Second Optimization
- **Monitor actual vs requested tokens**: Use `return_stats=True` to get accurate metrics
- **Optimal parameters**: Use the parameter sweep tool to find best settings
- **Temperature tuning**: Lower temperatures (0.2-0.4) often yield better speeds
- **Step optimization**: Balance between quality and speed with the `steps` parameter
- **Token efficiency**: Track the percentage of requested tokens actually generated

### Monitoring Tools
- **Quick performance check**: `python tokens_per_second_monitor.py`
- **Real-time monitoring**: Continuous tracking with customizable intervals
- **Performance history**: Track improvements over time
- **Parameter sweeps**: Automatically find optimal settings

### Large Prompt Optimization
- **Test context limits**: Use `python large_prompt_benchmark.py` to find maximum context
- **Performance scaling**: Understand how speed changes with prompt size
- **Memory management**: Monitor GPU usage with large contexts
- **Context efficiency**: Balance prompt length with generation quality
- **Progressive testing**: Start small and scale up to find optimal size

### For A100 GPU
- Use `torch_dtype=torch.bfloat16` for efficiency
- Set `device_map="auto"` for optimal placement
- Use batch processing for multiple queries
- Enable gradient checkpointing if needed

### Memory Management
- Call `torch.cuda.empty_cache()` between generations
- Use smaller `max_new_tokens` for memory-constrained scenarios
- Consider using `torch.cuda.memory_summary()` for monitoring

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `max_new_tokens`
   - Use `torch.cuda.empty_cache()`
   - Ensure no other processes are using GPU

2. **Model Loading Errors**:
   - Check internet connection
   - Verify HuggingFace token if needed
   - Ensure sufficient disk space

3. **Slow Performance**:
   - Verify GPU usage with `nvidia-smi`
   - Check if using bfloat16 precision
   - Ensure CUDA version compatibility

### GPU Monitoring
```bash
# Check GPU status
nvidia-smi

# Monitor in real-time
watch -n 1 nvidia-smi
```

## Advanced Usage

### Custom System Messages
```python
result = demo.generate_code(
    "Create a web scraper",
    system_message="You are an expert Python developer focused on web scraping."
)
```

### Parameter Tuning
- **Temperature**: Controls randomness (0.1-1.0)
- **Top-p**: Nucleus sampling (0.1-1.0)
- **Steps**: More steps = potentially better quality
- **Token per step**: Balance between quality and speed

## Contributing

Feel free to submit issues and enhancement requests!

## License

This example code is provided as-is for educational purposes. Please check the model's license on HuggingFace.

## References

- [DiffuCoder Paper](https://arxiv.org/abs/2506.20639)
- [HuggingFace Model Page](https://huggingface.co/apple/DiffuCoder-7B-cpGRPO)
- [Apple ML Research](https://github.com/apple/ml-diffucoder) 