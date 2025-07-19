# CogVLM Documentation Index

## 📚 Documentation Overview

This repository contains comprehensive documentation for the CogVLM visual language model. Choose the documentation that best fits your needs:

## 🚀 Quick Start

### For New Users
- **[Quick Start Guide](QUICK_START_GUIDE.md)** - Get up and running in minutes with practical examples
- **[API Documentation](API_DOCUMENTATION.md)** - Comprehensive guide with examples and usage instructions

### For Developers
- **[API Reference](API_REFERENCE.md)** - Detailed function signatures and parameter documentation
- **[README.md](README.md)** - Original project documentation with benchmarks and examples

## 📖 Documentation Structure

### 1. [Quick Start Guide](QUICK_START_GUIDE.md)
**Best for**: New users, getting started quickly, practical examples

**Covers**:
- Installation and setup
- Basic usage examples
- Image description tasks
- Visual question answering
- Visual grounding
- Multi-round conversations
- Fine-tuning basics
- Troubleshooting common issues

**Key Features**:
- Step-by-step tutorials
- Copy-paste code examples
- Task-specific parameter recommendations
- Error handling patterns
- Performance optimization tips

### 2. [API Documentation](API_DOCUMENTATION.md)
**Best for**: Understanding the full API, comprehensive usage, advanced features

**Covers**:
- Complete API overview
- Core model classes and methods
- Utility functions and processing
- Demo scripts and interfaces
- Fine-tuning pipeline
- Configuration options
- Template system
- Performance considerations

**Key Features**:
- Detailed explanations of all components
- Usage examples for each function
- Configuration guidelines
- Best practices
- Contributing guidelines

### 3. [API Reference](API_REFERENCE.md)
**Best for**: Developers, function signatures, parameter details

**Covers**:
- Complete function signatures
- Parameter types and defaults
- Return value specifications
- Class hierarchies
- Data types and structures
- Error handling patterns

**Key Features**:
- Type-annotated function signatures
- Parameter descriptions
- Return type specifications
- Inheritance hierarchies
- Error handling documentation

### 4. [README.md](README.md)
**Best for**: Project overview, benchmarks, research details

**Covers**:
- Project introduction and features
- Performance benchmarks
- Model architecture details
- Research methodology
- Citation information

## 🎯 Use Case Guides

### Image Description
```python
# Quick example from Quick Start Guide
response, history, _ = chat(
    image_path="image.jpg",
    model=model,
    text_processor=text_processor,
    img_processor=img_processor,
    query="Describe this image in detail",
    history=[],
    max_length=2048,
    temperature=0.7
)
```

### Visual Question Answering
```python
# VQA example with optimized parameters
response, history, _ = chat(
    image_path="image.jpg",
    model=model,
    text_processor=vqa_text_processor,
    img_processor=img_processor,
    query="How many people are in this image?",
    history=[],
    max_length=2048,
    temperature=0.3  # Lower for focused answers
)
```

### Visual Grounding
```python
# Grounding example with result parsing
response, history, _ = chat(
    image_path="image.jpg",
    model=grounding_model,
    text_processor=text_processor,
    img_processor=img_processor,
    query="Point to the red car in the image",
    history=[],
    max_length=2048,
    temperature=0.5
)

# Parse and visualize results
parse_response(image, response, "grounding_output.png")
```

## 🔧 Common Tasks

### Model Loading
```python
# Basic model loading
from models.cogvlm_model import CogVLMModel
import argparse

args = argparse.Namespace(
    deepspeed=None,
    local_rank=0,
    rank=0,
    world_size=1,
    model_parallel_size=1,
    mode='inference',
    fp16=False,
    bf16=True,
    skip_init=True,
    use_gpu_initialization=True,
    device='cuda'
)

model, model_args = CogVLMModel.from_pretrained("cogvlm-chat-v1.1", args=args)
model = model.eval()
```

### Processor Setup
```python
# Initialize processors
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor

tokenizer = llama2_tokenizer("lmsys/vicuna-7b-v1.5", signal_type="chat")
text_processor = llama2_text_processor_inference(tokenizer, max_length=2048)
img_processor = get_image_processor(224)
```

### Chat Function
```python
# Main chat function
from utils.chat import chat

response, history, _ = chat(
    image_path="image.jpg",
    model=model,
    text_processor=text_processor,
    img_processor=img_processor,
    query="Your question here",
    history=[],
    max_length=2048,
    temperature=0.8
)
```

## 📋 Available Models

| Model Name | Use Case | Resolution | Features |
|------------|----------|------------|----------|
| `cogvlm-chat-v1.1` | Conversations | 224px | Multi-round chat, detailed responses |
| `cogvlm-base-224` | Base tasks | 224px | Standard image understanding |
| `cogvlm-base-490` | High-res tasks | 490px | Better resolution, improved performance |
| `cogvlm-grounding-generalist` | Grounding | 224px | Visual grounding, object localization |
| `cogvlm-grounding-base` | Base grounding | 224px | Basic grounding capabilities |

## 🎛️ Parameter Guidelines

### Temperature Settings
- **0.3-0.5**: Factual answers, counting, precise tasks
- **0.6-0.8**: Balanced responses, general descriptions
- **0.8-0.9**: Creative responses, storytelling

### Max Length Settings
- **512**: Short answers, VQA tasks
- **1024**: Standard responses
- **2048**: Detailed descriptions, conversations

### Model Selection
- **Chat**: `cogvlm-chat-v1.1` for conversations
- **VQA**: Use VQA signal type with chat model
- **Grounding**: `cogvlm-grounding-generalist` for localization
- **High-res**: `cogvlm-base-490` for better detail

## 🚨 Troubleshooting

### Common Issues

1. **Memory Issues**
   - Use BF16 precision: `--bf16`
   - Enable model parallel: `--nproc-per-node=2`
   - Reduce batch size or image resolution

2. **Download Issues**
   - Set custom model directory: `export SAT_HOME=/path/to/models`
   - Use local tokenizer: `--local_tokenizer /path/to/tokenizer`

3. **Performance Issues**
   - Use appropriate image resolution (224px vs 490px)
   - Enable mixed precision training
   - Use model parallel for large models

### Getting Help

1. **Check the Quick Start Guide** for common solutions
2. **Review API Documentation** for detailed explanations
3. **Consult API Reference** for function signatures
4. **Check the original README** for project-specific information

## 📚 Additional Resources

### External Links
- [Paper](https://arxiv.org/abs/2311.03079) - Research paper
- [Hugging Face](https://huggingface.co/THUDM/cogvlm-chat-hf) - Model repository
- [ModelScope](https://www.modelscope.cn/models/ZhipuAI/CogVLM/summary) - Alternative download
- [Web Demo](http://36.103.203.44:7861/) - Online demonstration

### Code Examples
- `cli_demo.py` - Command-line interface
- `web_demo.py` - Web-based interface
- `finetune_demo.py` - Fine-tuning pipeline
- `evaluate_demo.py` - Evaluation script

## 🤝 Contributing

When contributing to the documentation:

1. **Follow existing structure** and formatting
2. **Add comprehensive examples** for new features
3. **Update all relevant documentation** files
4. **Test examples** before submitting
5. **Include type annotations** for new functions

## 📝 Documentation Maintenance

This documentation is maintained alongside the codebase. When updating:

1. **Update Quick Start Guide** for new features
2. **Revise API Documentation** for changed interfaces
3. **Modify API Reference** for new functions
4. **Test all examples** to ensure they work
5. **Update this index** if structure changes

---

**Last Updated**: November 2023  
**Version**: 1.1  
**Maintainer**: CogVLM Development Team