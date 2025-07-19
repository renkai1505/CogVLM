# CogVLM Quick Start Guide

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Image Description](#image-description)
4. [Visual Question Answering](#visual-question-answering)
5. [Visual Grounding](#visual-grounding)
6. [Multi-round Conversations](#multi-round-conversations)
7. [Fine-tuning](#fine-tuning)
8. [Troubleshooting](#troubleshooting)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Download Model

The model will be automatically downloaded on first use, or you can manually download from:
- [Hugging Face](https://huggingface.co/THUDM/cogvlm-chat-hf)
- [ModelScope](https://www.modelscope.cn/models/ZhipuAI/CogVLM/summary)

## Basic Usage

### Simple Image Description

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

# Load image
image_url = "https://github.com/THUDM/CogVLM/blob/main/examples/1.png?raw=true"
image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')

# Generate description
query = 'Describe this image'
inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])
inputs = {
    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
    'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
}

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=2048, do_sample=False)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(outputs[0])
    print(response)
```

### Using the Chat Function

```python
from utils.chat import chat
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor
from models.cogvlm_model import CogVLMModel
import argparse

# Initialize model
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

# Initialize processors
tokenizer = llama2_tokenizer("lmsys/vicuna-7b-v1.5", signal_type="chat")
text_processor = llama2_text_processor_inference(tokenizer, max_length=2048)
img_processor = get_image_processor(224)

# Chat with image
response, history, _ = chat(
    image_path="path/to/image.jpg",
    model=model,
    text_processor=text_processor,
    img_processor=img_processor,
    query="Describe this image in detail",
    history=[],
    max_length=2048,
    temperature=0.8
)

print(response)
```

## Image Description

### Detailed Image Description

```python
# For detailed descriptions, use the chat version
response, history, _ = chat(
    image_path="image.jpg",
    model=model,
    text_processor=text_processor,
    img_processor=img_processor,
    query="Provide a detailed description of this image, including all visible objects, people, actions, and the overall scene.",
    history=[],
    max_length=2048,
    temperature=0.7
)
```

### Creative Image Description

```python
# For creative descriptions
response, history, _ = chat(
    image_path="image.jpg",
    model=model,
    text_processor=text_processor,
    img_processor=img_processor,
    query="Write a creative story based on this image",
    history=[],
    max_length=2048,
    temperature=0.9
)
```

### Technical Image Analysis

```python
# For technical analysis
response, history, _ = chat(
    image_path="image.jpg",
    model=model,
    text_processor=text_processor,
    img_processor=img_processor,
    query="Analyze this image from a technical perspective, including composition, lighting, and visual elements.",
    history=[],
    max_length=2048,
    temperature=0.6
)
```

## Visual Question Answering

### Simple VQA

```python
# Use VQA version for short answers
vqa_tokenizer = llama2_tokenizer("lmsys/vicuna-7b-v1.5", signal_type="vqa")
vqa_text_processor = llama2_text_processor_inference(vqa_tokenizer, max_length=2048)

response, history, _ = chat(
    image_path="image.jpg",
    model=model,
    text_processor=vqa_text_processor,
    img_processor=img_processor,
    query="How many people are in this image?",
    history=[],
    max_length=2048,
    temperature=0.3  # Lower temperature for more focused answers
)
```

### Complex VQA

```python
# For complex questions
response, history, _ = chat(
    image_path="image.jpg",
    model=model,
    text_processor=text_processor,
    img_processor=img_processor,
    query="What emotions are the people in this image expressing, and what might have led to these emotions?",
    history=[],
    max_length=2048,
    temperature=0.8
)
```

### Counting and Detection

```python
# For counting tasks
response, history, _ = chat(
    image_path="image.jpg",
    model=model,
    text_processor=text_processor,
    img_processor=img_processor,
    query="Count all the cars in this image and describe their colors.",
    history=[],
    max_length=2048,
    temperature=0.5
)
```

## Visual Grounding

### Basic Grounding

```python
# Load grounding model
grounding_model, grounding_model_args = CogVLMModel.from_pretrained("cogvlm-grounding-generalist", args=args)
grounding_model = grounding_model.eval()

# Use grounding prompts
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

# Parse grounding results
from utils.parser import parse_response
parse_response(image, response, "grounding_output.png")
```

### Multiple Object Grounding

```python
response, history, _ = chat(
    image_path="image.jpg",
    model=grounding_model,
    text_processor=text_processor,
    img_processor=img_processor,
    query="Identify and point to all the people wearing hats in this image",
    history=[],
    max_length=2048,
    temperature=0.5
)
```

### Spatial Relationship Grounding

```python
response, history, _ = chat(
    image_path="image.jpg",
    model=grounding_model,
    text_processor=text_processor,
    img_processor=img_processor,
    query="Show me the object that is to the left of the table",
    history=[],
    max_length=2048,
    temperature=0.5
)
```

## Multi-round Conversations

### Conversation with Context

```python
# Initialize conversation
history = []

# First question
response, history, _ = chat(
    image_path="image.jpg",
    model=model,
    text_processor=text_processor,
    img_processor=img_processor,
    query="What do you see in this image?",
    history=history,
    max_length=2048,
    temperature=0.8
)

print("Model:", response)

# Follow-up question
response, history, _ = chat(
    image_path="image.jpg",
    model=model,
    text_processor=text_processor,
    img_processor=img_processor,
    query="Can you tell me more about the person on the left?",
    history=history,
    max_length=2048,
    temperature=0.8
)

print("Model:", response)
```

### Comparative Analysis

```python
# Compare multiple aspects
queries = [
    "Describe the overall scene",
    "What are the main colors in this image?",
    "How would you describe the mood or atmosphere?",
    "What might have happened just before this moment?"
]

history = []
for query in queries:
    response, history, _ = chat(
        image_path="image.jpg",
        model=model,
        text_processor=text_processor,
        img_processor=img_processor,
        query=query,
        history=history,
        max_length=2048,
        temperature=0.8
    )
    print(f"Q: {query}")
    print(f"A: {response}\n")
```

## Fine-tuning

### Basic Fine-tuning Setup

```python
from models.cogvlm_model import FineTuneTrainCogVLMModel
from utils.language import llama2_text_processor
from utils.vision import get_image_processor

# Initialize fine-tuning model
args = argparse.Namespace(
    # ... your args here
    use_lora=True,
    lora_rank=10,
    use_qlora=False,
    use_ptuning=False
)

model = FineTuneTrainCogVLMModel(args)

# Disable untrainable parameters
model.disable_untrainable_params()

# Create processors
tokenizer = llama2_tokenizer("lmsys/vicuna-7b-v1.5", signal_type="base")
text_processor = llama2_text_processor(tokenizer, max_length=2048)
img_processor = get_image_processor(224)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

### Custom Dataset Preparation

```python
def prepare_custom_dataset(image_paths, captions):
    dataset = []
    for img_path, caption in zip(image_paths, captions):
        # Process image
        image = Image.open(img_path).convert('RGB')
        processed_image = img_processor(image)
        
        # Process text
        processed_text = text_processor(caption, "")
        
        # Combine
        sample = {
            **processed_text,
            'vision': processed_image
        }
        dataset.append(sample)
    
    return dataset
```

## Advanced Usage

### Batch Processing

```python
def process_multiple_images(image_paths, query):
    responses = []
    for img_path in image_paths:
        response, _, _ = chat(
            image_path=img_path,
            model=model,
            text_processor=text_processor,
            img_processor=img_processor,
            query=query,
            history=[],
            max_length=2048,
            temperature=0.8
        )
        responses.append(response)
    return responses
```

### Custom Generation Parameters

```python
# Adjust generation parameters for different tasks
def generate_with_params(image_path, query, task_type="description"):
    if task_type == "description":
        params = {
            "max_length": 2048,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50
        }
    elif task_type == "vqa":
        params = {
            "max_length": 1024,
            "temperature": 0.3,
            "top_p": 0.7,
            "top_k": 20
        }
    elif task_type == "grounding":
        params = {
            "max_length": 512,
            "temperature": 0.5,
            "top_p": 0.8,
            "top_k": 10
        }
    
    response, history, _ = chat(
        image_path=image_path,
        model=model,
        text_processor=text_processor,
        img_processor=img_processor,
        query=query,
        history=[],
        **params
    )
    return response
```

### Error Handling

```python
def safe_chat(image_path, query, max_retries=3):
    for attempt in range(max_retries):
        try:
            response, history, _ = chat(
                image_path=image_path,
                model=model,
                text_processor=text_processor,
                img_processor=img_processor,
                query=query,
                history=[],
                max_length=2048,
                temperature=0.8
            )
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return f"Error: {e}"
            time.sleep(1)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```python
   # Reduce batch size or use gradient checkpointing
   model.gradient_checkpointing_enable()
   ```

2. **Model Download Issues**
   ```python
   # Set custom model directory
   import os
   os.environ['SAT_HOME'] = '/path/to/models'
   ```

3. **Tokenizer Issues**
   ```python
   # Use local tokenizer
   tokenizer = llama2_tokenizer("/path/to/local/tokenizer", signal_type="chat")
   ```

4. **Image Processing Issues**
   ```python
   # Ensure image is in correct format
   from PIL import Image
   image = Image.open("image.jpg").convert('RGB')
   ```

### Performance Optimization

```python
# Use mixed precision
model = model.half()

# Use model parallel for large models
torchrun --standalone --nnodes=1 --nproc-per-node=2 your_script.py

# Optimize memory usage
torch.cuda.empty_cache()
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug output
import torch
torch.set_printoptions(profile="full")
```

## Best Practices

1. **Choose the Right Model Version**
   - Use `cogvlm-chat-v1.1` for conversations
   - Use `cogvlm-grounding-generalist` for grounding tasks
   - Use `cogvlm-base-490` for better resolution

2. **Optimize Parameters**
   - Lower temperature (0.3-0.5) for factual answers
   - Higher temperature (0.8-0.9) for creative responses
   - Adjust max_length based on expected response length

3. **Handle Images Properly**
   - Always convert to RGB format
   - Resize large images before processing
   - Use appropriate image resolution

4. **Manage Memory**
   - Use BF16 precision when possible
   - Clear cache between batches
   - Use model parallel for large models

5. **Error Handling**
   - Always wrap API calls in try-catch blocks
   - Implement retry logic for network issues
   - Validate inputs before processing