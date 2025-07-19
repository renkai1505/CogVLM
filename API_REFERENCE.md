# CogVLM API Reference

## Table of Contents

1. [Model Classes](#model-classes)
2. [Utility Functions](#utility-functions)
3. [Processing Classes](#processing-classes)
4. [Demo Scripts](#demo-scripts)
5. [Fine-tuning Functions](#fine-tuning-functions)

## Model Classes

### CogVLMModel

**File**: `models/cogvlm_model.py`

**Inheritance**: `LLaMAModel`

#### Constructor
```python
def __init__(self, args, transformer=None, parallel_output=True, **kwargs)
```

**Parameters**:
- `args` (argparse.Namespace): Model configuration arguments
- `transformer` (optional): Pre-built transformer
- `parallel_output` (bool, default=True): Whether to use parallel output
- `**kwargs`: Additional keyword arguments

#### Class Methods

##### `from_pretrained(model_name, args, overwrite_args=None)`
```python
@classmethod
def from_pretrained(cls, model_name: str, args: argparse.Namespace, overwrite_args: dict = None) -> Tuple[CogVLMModel, argparse.Namespace]
```

**Parameters**:
- `model_name` (str): Name of pretrained model
- `args` (argparse.Namespace): Model arguments
- `overwrite_args` (dict, optional): Arguments to override

**Returns**: Tuple of (model, model_args)

##### `add_model_specific_args(parser)`
```python
@classmethod
def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser
```

**Parameters**:
- `parser` (argparse.ArgumentParser): Argument parser

**Returns**: Updated argument parser

#### Instance Methods

##### `forward(input_ids, vision_expert_mask, image_embed_mask, **kwargs)`
```python
def forward(self, input_ids: torch.Tensor, vision_expert_mask: torch.Tensor = None, image_embed_mask: torch.Tensor = None, **kwargs) -> torch.Tensor
```

**Parameters**:
- `input_ids` (torch.Tensor): Input token IDs
- `vision_expert_mask` (torch.Tensor, optional): Vision expert mask
- `image_embed_mask` (torch.Tensor, optional): Image embedding mask
- `**kwargs`: Additional keyword arguments

**Returns**: Model output tensor

### FineTuneTrainCogVLMModel

**File**: `models/cogvlm_model.py`

**Inheritance**: `CogVLMModel`

#### Constructor
```python
def __init__(self, args, transformer=None, parallel_output=True, **kw_args)
```

**Parameters**:
- `args` (argparse.Namespace): Model configuration arguments
- `transformer` (optional): Pre-built transformer
- `parallel_output` (bool, default=True): Whether to use parallel output
- `**kw_args`: Additional keyword arguments

#### Instance Methods

##### `disable_untrainable_params()`
```python
def disable_untrainable_params(self) -> None
```

Disables parameters that should not be trained during fine-tuning.

### FineTuneTestCogVLMModel

**File**: `models/cogvlm_model.py`

**Inheritance**: `CogVLMModel`

#### Constructor
```python
def __init__(self, args, transformer=None, parallel_output=True, **kw_args)
```

**Parameters**:
- `args` (argparse.Namespace): Model configuration arguments
- `transformer` (optional): Pre-built transformer
- `parallel_output` (bool, default=True): Whether to use parallel output
- `**kw_args`: Additional keyword arguments

## Utility Functions

### Chat Functions

**File**: `utils/chat.py`

#### `chat()`
```python
def chat(
    image_path: str,
    model: CogVLMModel,
    text_processor: llama2_text_processor_inference,
    img_processor: callable,
    query: str,
    history: List[Tuple[str, str]] = None,
    image: PIL.Image = None,
    max_length: int = 4096,
    top_p: float = 0.95,
    top_k: int = 5,
    temperature: float = 0.95,
    repetition_penalty: float = 1.0,
    invalid_slices: List = [],
    no_prompt: bool = False
) -> Tuple[str, List[Tuple[str, str]], Tuple[torch.Tensor, PIL.Image]]
```

**Parameters**:
- `image_path` (str): Path or URL to image
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

#### `process_image()`
```python
def process_image(
    image_path: str,
    img_processor: callable,
    image: PIL.Image = None
) -> Tuple[dict, PIL.Image]
```

**Parameters**:
- `image_path` (str): Path or URL to image
- `img_processor` (callable): Image processor function
- `image` (PIL.Image, optional): PIL image object

**Returns**: Tuple of (processed_image_dict, pil_image)

### Language Processing

**File**: `utils/language.py`

#### `llama2_tokenizer()`
```python
def llama2_tokenizer(tokenizer_path: str, signal_type: str = "base") -> LlamaTokenizer
```

**Parameters**:
- `tokenizer_path` (str): Path to tokenizer (default: "lmsys/vicuna-7b-v1.5")
- `signal_type` (str): Type of signal ("base", "chat", "vqa")

**Returns**: LlamaTokenizer with signal_type attribute

#### `llama2_text_processor_inference`

**Constructor**:
```python
def __init__(self, tokenizer: LlamaTokenizer, max_target_length: int = 2048, image_length: int = 1225)
```

**Parameters**:
- `tokenizer` (LlamaTokenizer): Tokenizer instance
- `max_target_length` (int): Maximum target sequence length
- `image_length` (int): Length of image embeddings

**Methods**:

##### `__call__(prompt)`
```python
def __call__(self, prompt: str = "") -> dict
```

**Parameters**:
- `prompt` (str): Input prompt

**Returns**: Dictionary with processed inputs

##### `history_to_prompt(query, history)`
```python
def history_to_prompt(self, query: str, history: List[Tuple[str, str]]) -> str
```

**Parameters**:
- `query` (str): Current query
- `history` (List[Tuple[str, str]]): Conversation history

**Returns**: Formatted prompt string

##### `replace_tags_with_empty(text)`
```python
def replace_tags_with_empty(self, text: str) -> str
```

**Parameters**:
- `text` (str): Input text

**Returns**: Text with special tags removed

#### `llama2_text_processor`

**Constructor**:
```python
def __init__(self, tokenizer: LlamaTokenizer, max_target_length: int = 2048, image_length: int = 1225)
```

**Parameters**:
- `tokenizer` (LlamaTokenizer): Tokenizer instance
- `max_target_length` (int): Maximum target sequence length
- `image_length` (int): Length of image embeddings

**Methods**:

##### `__call__(caption, prompt="")`
```python
def __call__(self, caption: str, prompt: str = "") -> dict
```

**Parameters**:
- `caption` (str): Caption text
- `prompt` (str, default=""): Prompt text

**Returns**: Dictionary with processed inputs

##### `history_to_prompt(query, history)`
```python
def history_to_prompt(self, query: str, history: List[Tuple[str, str]]) -> str
```

**Parameters**:
- `query` (str): Current query
- `history` (List[Tuple[str, str]]): Conversation history

**Returns**: Formatted prompt string

### Vision Processing

**File**: `utils/vision.py`

#### `get_image_processor()`
```python
def get_image_processor(image_size: int) -> callable
```

**Parameters**:
- `image_size` (int): Target image size

**Returns**: Image processor function

#### `BlipImageEvalProcessor`

**Constructor**:
```python
def __init__(self, image_size: int = 384, mean: tuple = None, std: tuple = None)
```

**Parameters**:
- `image_size` (int, default=384): Target image size
- `mean` (tuple, optional): Normalization mean
- `std` (tuple, optional): Normalization standard deviation

**Methods**:

##### `__call__(item)`
```python
def __call__(self, item: PIL.Image) -> torch.Tensor
```

**Parameters**:
- `item` (PIL.Image): Input image

**Returns**: Processed image tensor

#### `blip2_image_processor_func_with_inputs()`
```python
def blip2_image_processor_func_with_inputs(image_processor: callable, image: PIL.Image) -> dict
```

**Parameters**:
- `image_processor` (callable): Image processor function
- `image` (PIL.Image): Input image

**Returns**: Dictionary with processed image inputs

### Response Parsing

**File**: `utils/parser.py`

#### `parse_response()`
```python
def parse_response(img: PIL.Image, response: str, output_fn: str = 'output.png') -> None
```

**Parameters**:
- `img` (PIL.Image): Input image
- `response` (str): Model response
- `output_fn` (str, default='output.png'): Output file path

**Returns**: None (saves image to file)

#### `draw_boxes()`
```python
def draw_boxes(image: PIL.Image, boxes: List, texts: List, output_fn: str = 'output.png') -> None
```

**Parameters**:
- `image` (PIL.Image): Input image
- `boxes` (List): List of bounding box coordinates
- `texts` (List): List of text labels
- `output_fn` (str): Output file path

**Returns**: None (saves image to file)

#### `boxstr_to_boxes()`
```python
def boxstr_to_boxes(box_str: str) -> List[List[float]]
```

**Parameters**:
- `box_str` (str): String representation of bounding boxes

**Returns**: List of bounding box coordinates

#### `text_to_dict()`
```python
def text_to_dict(text: str) -> dict
```

**Parameters**:
- `text` (str): Input text with bounding box annotations

**Returns**: Dictionary mapping noun phrases to bounding boxes

## Processing Classes

### ImageMixin

**File**: `models/cogvlm_model.py`

**Inheritance**: `BaseMixin`

#### Constructor
```python
def __init__(self, args: argparse.Namespace)
```

**Parameters**:
- `args` (argparse.Namespace): Model arguments

#### Instance Methods

##### `word_embedding_forward()`
```python
def word_embedding_forward(self, input_ids: torch.Tensor, output_cross_layer: bool, **kw_args) -> torch.Tensor
```

**Parameters**:
- `input_ids` (torch.Tensor): Input token IDs
- `output_cross_layer` (bool): Whether to output cross layer
- `**kw_args`: Additional keyword arguments

**Returns**: Word embeddings tensor

### GLU

**File**: `models/cogvlm_model.py`

**Inheritance**: `nn.Module`

#### Constructor
```python
def __init__(self, args: argparse.Namespace, in_features: int)
```

**Parameters**:
- `args` (argparse.Namespace): Model arguments
- `in_features` (int): Input feature dimension

#### Instance Methods

##### `forward()`
```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Parameters**:
- `x` (torch.Tensor): Input tensor

**Returns**: Processed tensor

## Demo Scripts

### CLI Demo

**File**: `cli_demo.py`

#### `main()`
```python
def main() -> None
```

Main function for CLI demo.

**Command Line Arguments**:
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

**File**: `web_demo.py`

#### `load_model()`
```python
def load_model(args: argparse.Namespace) -> Tuple[CogVLMModel, callable, llama2_text_processor_inference]
```

**Parameters**:
- `args` (argparse.Namespace): Model arguments

**Returns**: Tuple of (model, image_processor, text_processor)

#### `post()`
```python
def post(
    input_text: str,
    temperature: float,
    top_p: float,
    top_k: int,
    image_prompt: str,
    result_previous: List,
    hidden_image: str
) -> Tuple[str, List, str]
```

**Parameters**:
- `input_text` (str): User input text
- `temperature` (float): Sampling temperature
- `top_p` (float): Top-p sampling
- `top_k` (int): Top-k sampling
- `image_prompt` (str): Image path
- `result_previous` (List): Previous results
- `hidden_image` (str): Hidden image data

**Returns**: Tuple of (input_text, result_text, hidden_image)

#### `main()`
```python
def main(args: argparse.Namespace) -> None
```

**Parameters**:
- `args` (argparse.Namespace): Command line arguments

## Fine-tuning Functions

**File**: `finetune_demo.py`

#### `disable_untrainable_params()`
```python
def disable_untrainable_params(self) -> None
```

Disables parameters that should not be trained during fine-tuning.

#### `data_collator()`
```python
def data_collator(examples: List[dict]) -> dict
```

**Parameters**:
- `examples` (List[dict]): List of training examples

**Returns**: Collated batch dictionary

#### `broadcast_auto()`
```python
def broadcast_auto(data_dict: dict) -> dict
```

**Parameters**:
- `data_dict` (dict): Data dictionary

**Returns**: Broadcasted data dictionary

#### `get_batch()`
```python
def get_batch(data_iterator: Iterator, args: argparse.Namespace, timers: callable) -> dict
```

**Parameters**:
- `data_iterator` (Iterator): Data iterator
- `args` (argparse.Namespace): Model arguments
- `timers` (callable): Timer function

**Returns**: Batch dictionary

#### `chat()`
```python
def chat(
    model: CogVLMModel,
    tokenizer: LlamaTokenizer,
    tokens: torch.Tensor,
    max_length: int = 1800,
    num_beams: int = 5,
    top_p: float = 0.95,
    top_k: int = 0,
    temperature: float = 0.8,
    **kwargs
) -> torch.Tensor
```

**Parameters**:
- `model` (CogVLMModel): Model instance
- `tokenizer` (LlamaTokenizer): Tokenizer instance
- `tokens` (torch.Tensor): Input tokens
- `max_length` (int, default=1800): Maximum sequence length
- `num_beams` (int, default=5): Number of beams for beam search
- `top_p` (float, default=0.95): Top-p sampling
- `top_k` (int, default=0): Top-k sampling
- `temperature` (float, default=0.8): Sampling temperature
- `**kwargs`: Additional keyword arguments

**Returns**: Generated token tensor

#### `forward_step()`
```python
def forward_step(data_iterator: Iterator, model: CogVLMModel, args: argparse.Namespace, timers: callable) -> Tuple[torch.Tensor, dict]
```

**Parameters**:
- `data_iterator` (Iterator): Data iterator
- `model` (CogVLMModel): Model instance
- `args` (argparse.Namespace): Model arguments
- `timers` (callable): Timer function

**Returns**: Tuple of (loss, metrics)

#### `forward_step_eval()`
```python
def forward_step_eval(data_iterator: Iterator, model: CogVLMModel, args: argparse.Namespace, timers: callable) -> Tuple[torch.Tensor, dict]
```

**Parameters**:
- `data_iterator` (Iterator): Data iterator
- `model` (CogVLMModel): Model instance
- `args` (argparse.Namespace): Model arguments
- `timers` (callable): Timer function

**Returns**: Tuple of (loss, metrics)

#### `create_dataset_function()`
```python
def create_dataset_function(
    image_processor: callable,
    text_processor: llama2_text_processor,
    path: str,
    args: argparse.Namespace
) -> callable
```

**Parameters**:
- `image_processor` (callable): Image processor function
- `text_processor` (llama2_text_processor): Text processor
- `path` (str): Dataset path
- `args` (argparse.Namespace): Model arguments

**Returns**: Dataset creation function

## Model Merging

**File**: `merge_model.py`

#### `main()`
```python
def main() -> None
```

Main function for merging fine-tuned models.

**Command Line Arguments**:
- `--version` (str): Model version
- `--bf16` (flag): Use BF16 precision
- `--from_pretrained` (str): Path to pretrained model

## Evaluation

**File**: `evaluate_demo.py`

#### `main()`
```python
def main() -> None
```

Main function for model evaluation.

**Command Line Arguments**:
- `--from_pretrained` (str): Path to pretrained model
- `--version` (str): Model version
- `--english` (flag): English-only output
- `--bf16` (flag): Use BF16 precision
- `--max_length` (int): Maximum sequence length
- `--top_p` (float): Top-p sampling
- `--top_k` (int): Top-k sampling
- `--temperature` (float): Sampling temperature

## Template System

**File**: `utils/template.py`

### Available Templates

#### `cn_template`
```python
cn_template: List[str]
```
List of Chinese image description prompts.

#### `en_template`
```python
en_template: List[str]
```
List of English image description prompts.

#### `en_template_q`
```python
en_template_q: List[str]
```
List of question-based English image description prompts.

## Data Types

### Common Types

```python
from typing import List, Tuple, Dict, Optional, Union, Callable
import torch
from PIL import Image
import argparse
from transformers import LlamaTokenizer
```

### Model Types

```python
from models.cogvlm_model import CogVLMModel, FineTuneTrainCogVLMModel, FineTuneTestCogVLMModel
from utils.language import llama2_text_processor, llama2_text_processor_inference
from utils.vision import get_image_processor, BlipImageEvalProcessor
from utils.chat import chat
from utils.parser import parse_response, draw_boxes, text_to_dict
```

## Error Handling

### Common Exceptions

1. **MemoryError**: Insufficient GPU memory
2. **FileNotFoundError**: Model or tokenizer not found
3. **ValueError**: Invalid model arguments
4. **RuntimeError**: CUDA/GPU related errors

### Error Recovery

```python
try:
    model, model_args = CogVLMModel.from_pretrained(model_name, args)
except Exception as e:
    print(f"Error loading model: {e}")
    # Handle error appropriately
```

## Performance Considerations

### Memory Usage

- Use `bf16` precision for reduced memory usage
- Enable model parallel for large models
- Use gradient checkpointing during training

### Speed Optimization

- Use appropriate image resolution (224px vs 490px)
- Enable mixed precision training
- Use model parallel inference for large models

### Batch Processing

```python
# Process multiple images efficiently
for batch in dataloader:
    outputs = model(**batch)
    # Process outputs
```