# CogVLM API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Core Models](#core-models)
4. [Utility Functions](#utility-functions)
5. [Demo Scripts](#demo-scripts)
6. [Fine-tuning](#fine-tuning)
7. [Examples](#examples)

## Overview

CogVLM is a powerful open-source visual language model (VLM) with 10 billion vision parameters and 7 billion language parameters. It achieves state-of-the-art performance on various cross-modal benchmarks and supports multiple tasks including image description, visual question answering, and visual grounding.

## Installation

### Prerequisites

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Hardware Requirements

- **Model Inference**: 1 × A100(80G) or 2 × RTX 3090(24G)
- **Fine-tuning**: 4 × A100(80G) [Recommended] or 8 × RTX 3090(24G)

## Core Models

### CogVLMModel

The main model class for CogVLM inference and training.

**Location**: `models/cogvlm_model.py`

#### Class Definition

```python
class CogVLMModel(LLaMAModel):
    def __init__(self, args, transformer=None, parallel_output=True, **kwargs):
        # Initialize CogVLM model
```

#### Key Methods

##### `from_pretrained(model_name, args, overwrite_args=None)`

Load a pretrained CogVLM model.

**Parameters**:
- `model_name` (str): Name of the pretrained model
  - `"cogvlm-chat-v1.1"`: Chat version with multi-round conversation support
  - `"cogvlm-base-224"`: Base model with 224px resolution
  - `"cogvlm-base-490"`: Base model with 490px resolution
  - `"cogvlm-grounding-generalist"`: Grounding model for visual grounding tasks
  - `"cogvlm-grounding-base"`: Base grounding model
- `args` (argparse.Namespace): Model configuration arguments
- `overwrite_args` (dict, optional): Arguments to override

**Returns**: Tuple of (model, model_args)

**Example**:
```python
import argparse
from models.cogvlm_model import CogVLMModel

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

model, model_args = CogVLMModel.from_pretrained(
    "cogvlm-chat-v1.1",
    args=args
)
```

##### `add_model_specific_args(parser)`

Add model-specific command line arguments.

**Parameters**:
- `parser` (argparse.ArgumentParser): Argument parser

**Returns**: Updated argument parser

**Added Arguments**:
- `--image_length` (int, default=256): Length of image embeddings
- `--eva_args` (dict, default={}): EVA model arguments

### FineTuneTrainCogVLMModel

Model class for fine-tuning with trainable parameters.

**Location**: `models/cogvlm_model.py`

#### Class Definition

```python
class FineTuneTrainCogVLMModel(CogVLMModel):
    def __init__(self, args, transformer=None, parallel_output=True, **kw_args):
        # Initialize fine-tuning model
```

#### Key Methods

##### `disable_untrainable_params()`

Disable parameters that should not be trained during fine-tuning.

**Supported Training Methods**:
- `--use_ptuning`: Prompt tuning
- `--use_lora`: LoRA fine-tuning
- `--use_qlora`: QLoRA fine-tuning

### FineTuneTestCogVLMModel

Model class for testing fine-tuned models.

**Location**: `models/cogvlm_model.py`

## Utility Functions

### Chat Function

**Location**: `utils/chat.py`

#### `chat(image_path, model, text_processor, img_processor, query, history=None, image=None, max_length=4096, top_p=0.95, top_k=5, temperature=0.95, repetition_penalty=1.0, invalid_slices=[], no_prompt=False)`

Main function for interacting with CogVLM model.

**Parameters**:
- `image_path` (str): Path or URL to the image
- `model` (CogVLMModel): Loaded CogVLM model
- `text_processor` (llama2_text_processor_inference): Text processor
- `img_processor` (callable): Image processor function
- `query` (str): User query/question
- `history` (List[Tuple[str, str]], optional): Conversation history
- `image` (PIL.Image, optional): PIL image object
- `max_length` (int, default=4096): Maximum sequence length
- `top_p` (float, default=0.95): Top-p sampling parameter
- `top_k` (int, default=5): Top-k sampling parameter
- `temperature` (float, default=0.95): Sampling temperature
- `repetition_penalty` (float, default=1.0): Repetition penalty
- `invalid_slices` (List, default=[]): Invalid token slices
- `no_prompt` (bool, default=False): Whether to use no prompt mode

**Returns**: Tuple of (response, history, (torch_image, pil_img))

**Example**:
```python
from utils.chat import chat
from utils.language import llama2_text_processor_inference
from utils.vision import get_image_processor

# Initialize processors
text_processor = llama2_text_processor_inference(tokenizer, max_length=2048)
img_processor = get_image_processor(224)

# Chat with model
response, history, (torch_image, pil_img) = chat(
    image_path="path/to/image.jpg",
    model=model,
    text_processor=text_processor,
    img_processor=img_processor,
    query="Describe this image",
    history=[],
    max_length=2048,
    temperature=0.8
)
```

### Language Processing

**Location**: `utils/language.py`

#### `llama2_tokenizer(tokenizer_path, signal_type="base")`

Create a tokenizer for CogVLM.

**Parameters**:
- `tokenizer_path` (str): Path to tokenizer (default: "lmsys/vicuna-7b-v1.5")
- `signal_type` (str): Type of signal ("base", "chat", "vqa")

**Returns**: LlamaTokenizer with signal_type attribute

#### `llama2_text_processor_inference(tokenizer, max_target_length=2048, image_length=1225)`

Text processor for inference.

**Parameters**:
- `tokenizer` (LlamaTokenizer): Tokenizer instance
- `max_target_length` (int): Maximum target sequence length
- `image_length` (int): Length of image embeddings

**Key Methods**:
- `__call__(prompt)`: Process text prompt
- `history_to_prompt(query, history)`: Convert query and history to prompt
- `replace_tags_with_empty(text)`: Remove special tags from text

### Vision Processing

**Location**: `utils/vision.py`

#### `get_image_processor(image_size)`

Create an image processor function.

**Parameters**:
- `image_size` (int): Target image size

**Returns**: Function that processes images for the model

**Example**:
```python
from utils.vision import get_image_processor

# Create image processor for 224px images
img_processor = get_image_processor(224)

# Process image
processed_image = img_processor(pil_image)
```

#### `BlipImageEvalProcessor`

Image processor class for evaluation.

**Parameters**:
- `image_size` (int, default=384): Target image size
- `mean` (tuple, optional): Normalization mean
- `std` (tuple, optional): Normalization standard deviation

### Response Parsing

**Location**: `utils/parser.py`

#### `parse_response(img, response, output_fn='output.png')`

Parse model response and draw bounding boxes for grounding tasks.

**Parameters**:
- `img` (PIL.Image): Input image
- `response` (str): Model response
- `output_fn` (str, default='output.png'): Output file path

**Example**:
```python
from utils.parser import parse_response

# Parse grounding response
parse_response(image, model_response, "grounding_result.png")
```

#### `draw_boxes(image, boxes, texts, output_fn='output.png')`

Draw bounding boxes and text on image.

**Parameters**:
- `image` (PIL.Image): Input image
- `boxes` (List): List of bounding box coordinates
- `texts` (List): List of text labels
- `output_fn` (str): Output file path

#### `text_to_dict(text)`

Extract noun phrases and bounding boxes from text.

**Parameters**:
- `text` (str): Input text with bounding box annotations

**Returns**: Dictionary mapping noun phrases to bounding boxes

## Demo Scripts

### CLI Demo

**Location**: `cli_demo.py`

Interactive command-line interface for CogVLM.

**Usage**:
```bash
# Chat version
python cli_demo.py --from_pretrained cogvlm-chat-v1.1 --version chat --english --bf16

# VQA version
python cli_demo.py --from_pretrained cogvlm-chat-v1.1 --version vqa --english --bf16

# Base model
python cli_demo.py --from_pretrained cogvlm-base-224 --version base --english --bf16 --no_prompt

# Grounding model
python cli_demo.py --from_pretrained cogvlm-grounding-generalist --version base --english --bf16
```

**Arguments**:
- `--max_length` (int, default=2048): Maximum sequence length
- `--top_p` (float, default=0.4): Top-p sampling
- `--top_k` (int, default=1): Top-k sampling
- `--temperature` (float, default=0.8): Sampling temperature
- `--english` (flag): English-only output
- `--version` (str, default="chat"): Model version
- `--from_pretrained` (str): Pretrained model name
- `--local_tokenizer` (str): Tokenizer path
- `--no_prompt` (flag): No prompt mode
- `--fp16` (flag): Use FP16 precision
- `--bf16` (flag): Use BF16 precision

### Web Demo

**Location**: `web_demo.py`

Web-based interface using Gradio.

**Usage**:
```bash
# Chat version
python web_demo.py --from_pretrained cogvlm-chat-v1.1 --version chat --english --bf16

# Grounding version
python web_demo.py --from_pretrained cogvlm-grounding-generalist --version base --english --bf16
```

**Features**:
- Interactive web interface
- Image upload support
- Conversation history
- Adjustable generation parameters
- Example inputs

### Multi-GPU Inference

For multi-GPU inference:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 cli_demo.py --from_pretrained cogvlm-chat-v1.1 --version chat --english --bf16
```

## Fine-tuning

### Fine-tuning Script

**Location**: `finetune_demo.py`

Complete fine-tuning pipeline for custom tasks.

**Usage**:
```bash
# Start fine-tuning
bash scripts/finetune_224_lora.sh
# or
bash scripts/finetune_490_lora.sh
```

**Key Functions**:

#### `forward_step(data_iterator, model, args, timers)`

Training forward step.

#### `forward_step_eval(data_iterator, model, args, timers)`

Evaluation forward step.

#### `create_dataset_function(image_processor, text_processor, path, args)`

Create dataset for fine-tuning.

### Model Merging

**Location**: `merge_model.py`

Merge fine-tuned model to single GPU format.

**Usage**:
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 merge_model.py --version base --bf16 --from_pretrained ./checkpoints/merged_lora_224
```

### Evaluation

**Location**: `evaluate_demo.py`

Evaluate fine-tuned model performance.

**Usage**:
```bash
bash scripts/evaluate_224.sh
# or
bash scripts/evaluate_490.sh
```

## Examples

### Basic Usage with Transformers

```python
import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

# Load model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
model = AutoModelForCausalLM.from_pretrained(
    'THUDM/cogvlm-chat-hf',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to('cuda').eval()

# Chat example
query = 'Describe this image'
image = Image.open(requests.get('https://github.com/THUDM/CogVLM/blob/main/examples/1.png?raw=true', stream=True).raw).convert('RGB')
inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])
inputs = {
    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
    'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
}
gen_kwargs = {"max_length": 2048, "do_sample": False}

with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0]))
```

### Custom Fine-tuning

```python
from models.cogvlm_model import FineTuneTrainCogVLMModel
from utils.language import llama2_text_processor
from utils.vision import get_image_processor

# Initialize model for fine-tuning
model = FineTuneTrainCogVLMModel(args)

# Disable untrainable parameters
model.disable_untrainable_params()

# Create processors
text_processor = llama2_text_processor(tokenizer, max_length=2048)
img_processor = get_image_processor(224)

# Training loop
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

### Grounding Task

```python
# Load grounding model
model, model_args = CogVLMModel.from_pretrained("cogvlm-grounding-generalist", args=args)

# Use grounding prompts
query = "Point to the red car in the image"
response, history, _ = chat(
    image_path="image.jpg",
    model=model,
    text_processor=text_processor,
    img_processor=img_processor,
    query=query
)

# Parse grounding results
parse_response(image, response, "grounding_output.png")
```

## Template System

**Location**: `utils/template.py`

The template system provides various prompts for different tasks:

### Image Description Templates

- `cn_template`: Chinese image description prompts
- `en_template`: English image description prompts
- `en_template_q`: Question-based image description prompts

### Usage

```python
from utils.template import en_template
import random

# Select random template
template = random.choice(en_template)
query = template + " [IMAGE]"
```

## Configuration

### Model Arguments

Key configuration parameters:

- `--image_length`: Length of image embeddings (default: 256)
- `--eva_args`: EVA model configuration
- `--hidden_size`: Hidden layer size
- `--num_layers`: Number of transformer layers
- `--num_attention_heads`: Number of attention heads

### Training Arguments

- `--pre_seq_len`: Prompt tuning sequence length
- `--lora_rank`: LoRA rank for fine-tuning
- `--use_ptuning`: Enable prompt tuning
- `--use_lora`: Enable LoRA fine-tuning
- `--use_qlora`: Enable QLoRA fine-tuning
- `--layer_range`: Layer range for fine-tuning

## Troubleshooting

### Common Issues

1. **Memory Issues**: Use model parallel inference or reduce batch size
2. **Tokenizer Issues**: Ensure correct tokenizer path is specified
3. **Download Issues**: Use local tokenizer or download from alternative sources
4. **Grounding Issues**: Ensure correct prompt format for grounding tasks

### Environment Variables

- `SAT_HOME`: Set custom model download directory
- `RANK`: Distributed training rank
- `WORLD_SIZE`: Distributed training world size

## Performance Tips

1. **Use BF16**: Enable BF16 for better performance and memory efficiency
2. **Model Parallel**: Use model parallel for large models
3. **Image Resolution**: Use 490px resolution for better performance
4. **Batch Processing**: Process multiple images in batches when possible

## Contributing

When contributing to CogVLM:

1. Follow the existing code structure
2. Add comprehensive documentation for new functions
3. Include examples for new features
4. Test with different model configurations
5. Update this documentation for new APIs