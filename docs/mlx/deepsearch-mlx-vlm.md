Great, I’ll prepare a comprehensive Markdown guide for `mlx-vlm`, targeting expert Python users who want to integrate it into their own libraries (like AbstractLLM), with a focus on MLX-based workflows. The guide will cover:

* How `mlx-vlm` works and its limitations
* Local usage and API integration
* Concrete examples for adding models like Kimi-VL, Qwen2.5-VL, Llama-4-Scout, etc.
* How to wire this into another Python application using modular interfaces

I'll get started and update you shortly with the complete guide.


# Technical Guide: Understanding and Integrating **MLX-VLM** (Vision-Language Models on Apple Silicon)

## Overview of **MLX-VLM**

**MLX-VLM** is a specialized Python library for running vision-language models (VLMs) natively on Apple Silicon Macs. It leverages Apple’s **MLX** framework – an array-based machine learning library optimized for Apple’s M-series GPUs – instead of PyTorch or CUDA. In essence, MLX serves a similar role to NumPy/PyTorch but is *exclusive to Apple Silicon*. This means MLX-VLM can only run models that have been **converted to MLX’s format** (usually via the community or MLX tools). Models not in MLX format (e.g. standard PyTorch `.bin` weights) are **not supported** without conversion, and the library does not use PyTorch at all in its execution pipeline.

**Architecture & Design Principles:** MLX-VLM is designed to maximize performance on Macs by utilizing the Metal GPU and the **unified memory architecture** of Apple Silicon. Unlike traditional frameworks which often transfer data between CPU and GPU memory, MLX uses unified memory so that both the image data and model weights reside in the same memory space. This design eliminates costly CPU-GPU copies and significantly reduces latency for vision tasks. The architecture of a VLM in MLX-VLM typically consists of a **visual encoder** (e.g. a vision transformer or CNN that processes images) and a **language decoder** (a large language model that generates text). MLX-VLM integrates these components into a single model graph optimized for Metal. All computations – from image feature extraction to autoregressive text generation – execute on the Apple Neural Engine or GPU via Metal, with minimal Python overhead.

**Limitations:** Because MLX-VLM is tightly coupled to Apple’s MLX, it runs **only on Apple Silicon hardware** (M1, M2, M3 chips, etc.) and **macOS**. There is no CUDA or Windows/Linux support. Models must be pre-converted to MLX format (usually provided through the community on Hugging Face) – you cannot directly load a PyTorch model checkpoint. Additionally, only certain model architectures are implemented in MLX-VLM; support for new models may lag behind their open-source releases (until the community or library author adds them). For example, as new vision models like Qwen’s “QVQ” or Google’s Gemma are released, MLX-VLM may require an update to support their architecture. MLX-VLM focuses on **inference and fine-tuning** via parameter-efficient methods (LoRA/QLoRA), but does not support full PyTorch-style training of these models. Despite these limitations, the design ethos is to provide **efficient local execution** of large VLMs with minimal setup: just `pip install mlx-vlm` and you can run multi-billion-parameter image+text models on your Mac.

## Supported Models in MLX-VLM

MLX-VLM supports a growing list of multimodal models that have been converted to MLX format. The models are hosted on Hugging Face under the `mlx-community` organization (each model repo typically contains MLX-optimized safetensor weights, configuration, and chat templates). Some of the currently supported vision-language models include:

* **Alibaba Qwen-VL series:** *Qwen2-VL* (second-gen Qwen vision models, e.g. 2B and 7B parameters) and *Qwen2.5-VL* (improved 2.5 generation, available in 3B, 7B, and a massive 72B variant). These are instruction-tuned multimodal models by Alibaba. MLX-VLM provides them in 4-bit, 6-bit, or 8-bit quantized form for feasible on-device use (e.g. `Qwen2.5-VL-72B-Instruct-6bit`). Qwen models are state-of-the-art open VLMs with conversation capabilities.

* **LLaVA and LLaMA-Vision variants:** Models like LLaVA (Large Language and Vision Assistant) and vision-enabled LLaMA derivatives are supported. For example, the **LLaMA 3.2 Vision** series (community models that add vision encoders to LLaMA 3.2 in 11B and 90B sizes) have MLX conversions. LLaVA-style models allow image-question answering by combining CLIP vision encoders with LLaMA-based decoders. An “Interleaved” version of LLaVA is supported for multi-image inputs.

* **IDEFICS:** IDEFICS is an 80B-parameter open multimodal model (Flamingo-style) from Hugging Face. MLX-VLM supports IDEFICS in its 80B form and smaller distilled versions (e.g. \~9B). It is designed for image-to-text generation with strong performance. Notably, MLX-VLM can run **Idefics 80B** on high-memory Macs by using 4-bit quantization. IDEFICS and its fine-tuned variant (sometimes referenced as IDEFICS 2 or 3 in community discussions) support analyzing multiple images in one query.

* **Pixtral 12B:** Pixtral is a 12B parameter vision-enabled model based on Mistral AI’s work. It’s known for strong multimodal reasoning at a relatively smaller size. MLX-VLM supports Pixtral (quantized to 4bit or 8bit) and it has been showcased as running efficiently on Apple Silicon.

* **Microsoft Phi-DAE models:** *Phi-3* and *Phi-3.5 Vision* models (from Microsoft’s Project Florence) are supported in instruct form. These are smaller (3–4 billion parameter) vision-language models, useful for mobile deployment. For instance, **Phi-3.5 Vision Instruct** (a 4-bit quantized model with \~1B parameters) can run even on iOS devices via MLX. They are not as conversationally fluent as larger models but are fast and efficient.

* **Google PaLI/Gemma family:** **PaliGemma 2** mix models (Gemma 2) have been integrated. These are Google’s vision models (3B, 10B, 28B sizes) capable of OCR, captioning, and visual reasoning. For example, `paligemma-3b-mix-224-8bit` is available (3B parameters, 224×224 image resolution, 8-bit). Gemma 3 (27B) is a newer Google model – at time of writing, an MLX conversion may exist but integrating it fully is a work in progress (we’ll discuss how to extend support later).

* **Community models:** Many community-contributed VLMs are in MLX format. For example, **Kimi-VL-A3B** (Moonshot AI’s model built on InternVL + InternLM, \~1B params) is available in 4-bit and 6-bit MLX form. Also, models like **Dolphin-Vision-72B** (an uncensored multimodal 72B model) and **Aya 8B/32B Vision** (from Cohere’s dataset) have MLX versions. Even experimental models like **Qwen QVQ-72B-Preview** (focusing on step-by-step visual reasoning) can be converted for MLX, though official support may lag.

All these models are distributed via Hugging Face Hub under the `mlx-community` namespace, which serves as a repository of *pre-converted, ready-to-run models*. Each model repository usually indicates the original model it was converted from and the MLX-VLM version used for conversion (for example, the MLX model card might note it was converted from an original HuggingFace model using `mlx-vlm v0.1.x`). The **model files** are in `safetensors` format (for efficient, safe loading) and typically quantized (4-bit, 6-bit, 8-bit, or sometimes mixed precision BF16) to balance memory usage and performance. These quantized, MLX-optimized weights often allow running models that would otherwise be too large – e.g., a 72B model at 6-bit can fit in tens of GB of memory instead of hundreds. Keep in mind that to use any of these, you simply reference the `mlx-community/Model-Name` in the MLX-VLM API and the library will handle downloading and loading the appropriate files.

**Note:** Not every vision model on Hugging Face is immediately usable – only those that have been converted to MLX. If a model you want is not yet in `mlx-community`, you would need to convert it (or request the community to do so). We will cover how to add new models in a later section.

## How **MLX-VLM** Works Internally

Under the hood, MLX-VLM uses Apple’s MLX framework to load model weights and perform inference in a way that is optimized for the Mac’s hardware. Let’s break down the internal steps:

* **Model Loading and Initialization:** When you call `mlx_vlm.load(model_path)`, the library will fetch the model files from the Hugging Face Hub (if not already cached). This uses the Hugging Face Python API or `huggingface_hub` under the covers to download the `safetensors` weight shards and any config files (like `config.json`, `tokenizer.json`, `vocab.txt`, etc.) associated with the model. Each model has an architecture type, specified either by naming conventions or config entries. MLX-VLM uses this to determine how to reconstruct the model in MLX. For example, a Qwen-VL model will be assembled using a Qwen transformer decoder and whatever image encoder it expects. **No PyTorch** is used at runtime – instead MLX-VLM uses **MLCompute/Metal operations** via the MLX library to allocate tensors and layers. In essence, MLX-VLM builds a computational graph similar to how one would in PyTorch or TensorFlow, but using MLX’s API (which is designed for Apple’s GPU). The model’s weights (once downloaded) are loaded into MLX tensor arrays (on device memory). At this stage, MLX might apply certain optimizations, like converting weights to lower precision if needed and setting up the model for FastFP16 or int8 matrix ops on the GPU.

* **Preprocessing Inputs (Images & Prompt):** Before inference, any image inputs must be preprocessed into the format the model expects. MLX-VLM provides a **processor** object (often returned alongside the model from `load`) that handles this. Typically, preprocessing includes: loading the image (if a filepath or URL is provided, MLX-VLM will load it via PIL under the hood – the README even shows you can pass a `PIL.Image.Image` directly), resizing and/or cropping the image to the target resolution, converting it to the model’s required color space (usually RGB), and normalizing pixel values. For most vision backbones (e.g. CLIP or ViT), images are resized to a square (224×224, 448×448, etc. depending on model) and normalized to a zero-centered range. The processor uses the info from the model’s `preprocessor_config.json` (if provided in the model repo) or defaults for that architecture. If a model expects multiple images as input, you can pass a list of image paths – MLX-VLM will process each and stack them appropriately. Multi-image models use techniques like concatenating image tokens or interleaving text and image tokens. MLX-VLM’s multi-image support extends to select models that were trained for it (e.g. IDEFICS, Qwen-VL, Pixtral can analyze several images jointly). In such cases, the processor will not only preprocess each image but also help format the prompt to indicate multiple images (some models expect special tokens like `<image1>` `<image2>` in the prompt).

* **Prompt Formatting and Tokenization:** Vision-language models often require a specific **prompt template** for best results – for instance, there might be a system instruction, or a special token where the image is “inserted”. MLX-VLM includes utility functions (e.g. `prompt_utils.apply_chat_template`) to apply a model-specific template to your prompt. It uses a JSON template (provided in the model repo, like `chat_template.json`) that defines how to wrap the user prompt, how to represent the image, and how the assistant should respond. For example, a template might look like:

  ```json
  {
    "system": "You are a helpful vision assistant.",
    "roles": ["user", "assistant"],
    "images": "<Image>",
    "sep": "\n"
  }
  ```

  The `apply_chat_template` function will combine the system prompt, your user prompt, and placeholders for images into a single formatted string. Then, MLX-VLM will **tokenize** this combined prompt using the model’s tokenizer (which is typically loaded as part of the processor). Tokenization converts the text (including any special tokens for images) into input IDs that the model can consume. For some architectures, the image is represented by a special token (e.g. `<Image>` or similar) which instructs the model to incorporate the image embeddings at that position. Other architectures might simply expect that image features are fed as “prefix” tokens with no explicit placeholder in text. MLX-VLM’s template system covers these cases by adjusting the prompt format and internally ensuring the image tensor is fed to the model in the correct way corresponding to the token positions.

* **Generation Pipeline:** Once the input is prepared (image tensors ready, text prompt tokenized), the **generation** begins. This is handled by `mlx_vlm.generate()`, which takes the model, processor, formatted prompt (as text or token IDs), and the preprocessed image tensor. Generation is done autoregressively: the model will produce output tokens one by one until it hits a stop condition (like an end-of-sequence token or reaching a max token limit). Internally, this loop is happening in MLX’s C++/Metal backend for efficiency. The MLX framework is optimized for *token-by-token* inference on Apple GPUs, meaning it can handle the causal attention and sampling on device. During generation, you can control parameters like `max_tokens` (maximum length of response), `temperature`, `top_p`, etc., similar to Hugging Face Transformers, either via a config or arguments to `generate`. The MLX-VLM generate function is non-streaming by default (it will compute the full output then return), but it’s fast. In a server scenario, you might use a streaming generate (if provided or by manually slicing tokens). Notably, MLX’s execution here benefits from **Apple’s Metal Performance Shaders** and the neural engine: it uses 16-bit floats or lower precision matrix multiplies that are accelerated by the hardware. The unified memory means the image data is already in place for the GPU to use – no extra copying needed. As each token is generated, the new token’s embedding is fed back in until completion. Despite the heavy computation (especially for multi-billion param models), Apple’s GPUs and **MLCompute** graph optimizations make this reasonably fast on device. For instance, a 2B parameter Qwen-VL model can run quite smoothly on an M1 Mac, and even larger models like 7B or 13B run decently with 4-bit quantization.

* **Output Processing:** The raw output from generation is a sequence of token IDs which MLX-VLM then decodes using the tokenizer to get the final text. Any post-processing (such as stripping prompt echoes or special tokens) is done according to the chat template or model config. The result is returned as a Python string (or can be a stream of tokens if using a streaming interface).

Behind the scenes, MLX-VLM’s integration with Apple’s hardware means it can also leverage features like memory “wiring” on macOS. For very large models that approach RAM limits, MLX can mark memory as wired (unswappable) to improve stability (this is handled automatically if macOS 15+ is detected). The key takeaway is that MLX-VLM abstracts away the low-level details – from your perspective, you load the model and call `generate`, and the library orchestrates all steps (image pre-processing, encoding, decoding, GPU execution) to produce the answer.

## Using **MLX-VLM** Locally: Installation and Example

Getting started with MLX-VLM on a local Mac is straightforward:

**Installation:** First, ensure you have a Mac with an M1, M2, or later Apple Silicon chip, and Python 3.8+. It’s recommended to use a Python virtual environment. Then install MLX-VLM via pip:

```bash
pip install mlx-vlm
```

This will pull the `mlx_vlm` package from PyPI (version 0.1.25 as of writing). The installation will also bring in the core `mlx` library as a dependency, plus any needed tokenizers. You do **not** need PyTorch or TensorFlow at all.

**Local Inference Example:** Below is a complete Python example of loading a vision-language model and generating text from an image and prompt. We’ll use a large Qwen 2.5 VL model in this example:

```python
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# 1. Load a vision-language model (weights will download if not cached)
model_name = "mlx-community/Qwen2.5-VL-72B-Instruct-6bit"  # 72B param Qwen2.5, 6-bit quantized
model, processor = load(model_name)
config = load_config(model_name)  # loads model config and template info

# 2. Prepare inputs: an image (or list of images) and a text prompt
image_path = "example.jpg"  # path to an image file
# You could also use a PIL image: e.g., Image.open("example.jpg") instead of a path
prompt = "What is happening in this image?"

# 3. Format the prompt using the model's chat template (adds system/user tokens, etc.)
formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)

# 4. Generate the model's response
output_text = generate(model, processor, formatted_prompt, [image_path], max_tokens=100)
print("Model output:", output_text)
```

Let’s break down what this does. The `load()` function returns a `model` (which is an MLX-accelerated model object) and a `processor`. The `processor` knows how to handle the image and tokenization for this specific model. We then load a `config` which contains additional settings (like special tokens or the chat template JSON).

In step 2, we specify our image and prompt. MLX-VLM allows passing the image either as a file path (string) or as a PIL `Image` object – internally it will convert it appropriately. We then call `apply_chat_template` to construct the final text prompt that the model expects. This inserts any required system prompt and placeholders for the image. We pass `num_images=1` since we have one image (if we had a list of images, this ensures the template knows to expect multiple image slots). For a Qwen model, for instance, this might produce a prompt like:

```
System: You are a helpful assistant.
User: <Image> What is happening in this image?
Assistant:
```

Finally, we call `generate()` with the model, processor, the formatted prompt, and the list of images. Under the hood, this will preprocess the image, tokenize the prompt (taking into account the image token), run the inference, and return the generated answer. We specified `max_tokens=100` to limit the length of the answer (and we could also set parameters like `temperature=0.2` or others as needed). The result printed is the assistant’s description of the image.

**Example Output:** If the image was, say, a photo of people hiking a mountain trail, the model might output: *"It looks like a group of hikers trekking up a rocky mountain path amidst a scenic landscape."*

This local example demonstrates that with just a few lines of code you can load a massive vision-language model and query it – all locally with no external API. The pipeline is straightforward thanks to MLX-VLM’s high-level functions. (The code above mirrors the official usage example, which shows using a 4-bit Qwen2-VL to describe an image.)

**Tips:** When running locally, be mindful of memory. Large models like the 72B Qwen in 6-bit require a lot of RAM (likely 64GB or more). If you only have 16GB or 32GB RAM, opt for a smaller model (e.g., `Qwen2.5-VL-7B-Instruct-4bit`) or a lower quantization. The good news is MLX-VLM will gracefully swap or use macOS’s memory compression if needed (though performance might suffer). For most use-cases, a 7B or 13B model at 4-bit gives a good balance. Also, ensure you run on a device with the Metal GPU (if you run headless via SSH, you may need to set `METAL_DEVICE_WRAPPER_TYPE=1` environment variable to use GPU).

MLX-VLM also provides a **CLI** and a Gradio-based UI for quick testing. For instance, after installation you could run:

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --prompt "Describe the image." --image example.jpg
```

to get an output on the command line. Or `python -m mlx_vlm.chat_ui --model mlx-community/Pixtral-12B-4bit` to launch an interactive chat browser. These are convenient for experimentation, but for integration into applications, you’ll use the Python API as shown above.

## Integrating MLX-VLM into a Python Service or Library

MLX-VLM is designed to be used as a module, which makes integration into larger applications (such as an AI service, or a framework like **AbstractLLM**) quite feasible. Here we’ll discuss how to wrap MLX-VLM functionality into your own classes and some architectural considerations for using it in production systems.

**Basic Integration Approach:** If you have an abstract interface for language models (for example, an `AbstractLLM` class with methods like `generate(prompt)` or `chat(history)`), you can implement a subclass that uses MLX-VLM under the hood. The key is to manage the model and processor as persistent objects (to avoid re-loading weights on every query) and simply call MLX-VLM’s functions when a request comes in.

For example, suppose your library expects a class with a `generate` method:

```python
class VisionLLM:  # This could inherit from your AbstractLLM base class
    def __init__(self, model_path: str):
        # Load the MLX-VLM model once during initialization
        self.model, self.processor = load(model_path)
        self.config = load_config(model_path)
        self.model_name = model_path

    def generate(self, prompt: str, images: list = None, **gen_kwargs) -> str:
        # Prepare images and prompt, then generate a response.
        if images is None:
            images = []
        elif isinstance(images, str) or not hasattr(images, '__iter__'):
            images = [images]  # accept a single image path or object
        # Format prompt with the appropriate template (if model uses images in prompt)
        formatted = apply_chat_template(self.processor, self.config, prompt, num_images=len(images))
        # Call MLX-VLM generate with any generation kwargs (e.g. temperature)
        output = generate(self.model, self.processor, formatted, images, **gen_kwargs)
        return output
```

In this snippet, the `VisionLLM` class encapsulates an MLX-VLM model. We load the model in the constructor (which might take some time for large models, so in a real service you might want to do this at application startup). The `generate` method then accepts a prompt and an optional list of images. We ensure the images are in a list form, apply the chat template, and then call `generate`. This returns the model’s text output, which we then return to the caller.

You can expand this to handle chat history: for example, if you want a multi-turn conversation, you could maintain a history of prompts and feed them all to the model (most MLX-VLM templates are designed for single-turn Q\&A, but you can concatenate or use the `system` role for context). Another approach is to use MLX-VLM’s built-in chat interface in code – but currently the Python API doesn’t maintain a persistent chat history for you, so you manage it externally by prepending previous Q\&A turns in the prompt.

**Integration with Web Services:** If building a web service (say with FastAPI or Flask), you would typically load the model at server start (perhaps in a global or a singleton object). Because MLX-VLM models can be large, you’ll likely want only one model loaded per process (the library even caches only one model at a time in its FastAPI server mode). If you need to handle multiple models (e.g., different endpoints for different models), be aware of memory constraints – loading two 13B models will require \~2× memory. You might consider multiple processes or dynamically unloading one model to load another if that scenario arises (MLX-VLM provides a `/unload` endpoint in its server for this purpose).

**Threading and Concurrency:** The MLX-VLM model object is not explicitly documented as thread-safe. Typically, you should serialize access to the `generate` call for a given model (one inference at a time per model instance). If you need to handle many requests in parallel, you could run multiple processes each with its own model (since Apple GPUs may not multi-task well on a single large model inference anyway). For many applications, running one request at a time is sufficient if the model responds in a couple of seconds. If you need asynchronous operation, you could wrap the `generate` call in a thread or `asyncio.to_thread` since the heavy work is in C++/Metal.

**Dependency and Packaging:** To include MLX-VLM in your project, add `mlx-vlm` to your `requirements.txt` or setup.py. Note that `mlx-vlm` will pull in the core `mlx` package (which includes native code). Ensure your deployment environment is an Apple Silicon Mac; if you use CI/CD, you might need to build on Mac or at least skip MLX-VLM installation on other OS. If your library (like AbstractLLM) is cross-platform, you should conditionally import or use MLX-VLM – e.g., check for platform and availability of Metal, otherwise throw an error or fallback to a CPU-based model. In a managed server scenario (like deploying to a Mac Mini server), make sure you have permissions to use the GPU (some virtualization containers might not allow it, so direct Mac deployment is easiest).

**Example: Integration in a FastAPI service** – MLX-VLM actually provides a simple FastAPI server out of the box. You can see in the README an example where they start a server and then send a JSON request with model, prompt, image, etc., and get a streamed response. You can mimic that in your own service: parse incoming requests to get an image (as file upload or URL) and a prompt, feed it to your `VisionLLM.generate` method, and return the result. If streaming output token-by-token is needed, you might integrate with an event/streaming mechanism; otherwise returning the full generated text is simpler.

**Architectural Considerations:** Running large MLX models can use a lot of memory and CPU/GPU. It’s wise to monitor resource usage. Apple’s instruments or `metalperf` can help profile GPU usage. Also consider the **warm-up time** – the first inference may be slightly slower as kernels compile; subsequent inferences are faster. You might do a dummy run on startup (e.g., run the model on a small prompt) to warm up. For long-running services, ensure macOS doesn’t nap the process (usually not an issue with continuous load).

In summary, integrating MLX-VLM is mostly a matter of wrapping the load and generate calls in your own interface. The heavy lifting (model management, optimization) is handled by MLX-VLM, so you can focus on application logic. Many developers have built wrappers like this; for instance, **Simon Willison’s LLM CLI** has an MLX integration that allows switching a backend to MLX-VLM with minimal changes. This shows that MLX-VLM can slot into abstracted LLM frameworks relatively easily.

## Extending MLX-VLM with New Models

As the field of multimodal AI evolves, new vision-language models are released frequently. If you want to use a model that **is not yet officially supported** by MLX-VLM, you have a few options to integrate it. We’ll walk through how to add a new model – from obtaining the weights to writing the code that makes it runnable. Let’s use a hypothetical example of adding **Google Gemma-3 27B** (a new model) and also discuss **Kimi-VL-A3B** and **Qwen QVQ-72B** as case studies.

**1. Obtain or Convert the Model Weights:** The first step is to get the model in **MLX format**. As mentioned, MLX only works with converted models. Check the Hugging Face `mlx-community` organization to see if the model has already been converted by someone. For example, for *Kimi-VL-A3B*, searching reveals `mlx-community/Kimi-VL-A3B-Thinking-6bit` which indicates it’s available (converted from the original moonshotai model). If your target model is there (perhaps as `mlx-community/Gemma3-27B-Instruct-4bit` for Gemma, or `mlx-community/QVQ-72B-Preview-8bit` for Qwen’s QVQ), you can use that path directly in `load()`. If not, you will need to convert it yourself using MLX tools.

To convert, you can use the **MLX conversion CLI**. For LLMs, the `mlx_lm.convert` command is used to quantize and convert a Hugging Face model to MLX. For VLMs, MLX-VLM might have similar functionality (often the conversion process for multimodal merges the visual encoder and language model weights). If no dedicated tool is provided, an approach is:

* Load the original model in PyTorch (if possible) or obtain its weights files.
* Use MLX’s Python API to create an MLX model and load weights into it.
* Serialize the MLX model to disk or directly upload to HF.

This process is non-trivial and may require writing some code in the MLX-VLM repository. A simpler method: sometimes the community provides a **conversion Space** on Hugging Face (like `mlx-my-repo` Space mentioned in the MLX docs) to do conversion in the browser. If available, you could input the original model repo and get an MLX model out.

For our purpose, let’s assume we now have the MLX weights available (either via community or conversion). We’ll proceed to integration.

**2. Create/Update the Processor for the Model:** Each model might have specific preprocessing needs. Most multimodal models use similar image preprocessing (e.g. normalize to Imagenet mean/std, specific resolution), but some might differ (maybe a model expects grayscale, or a different cropping). Check the original model’s documentation for how it processes images. For instance, Google’s Gemma might expect 224×224 images scaled to \[-1,1]. If MLX-VLM doesn’t already handle this, you’d implement a custom processor. This could mean writing a Python function or class that does the following:

* Resize image to the required size (maybe given in the model config or original repo).
* Normalize pixel values (some models use 0-1, others -1 to 1, others subtract specific means).
* Possibly apply other transforms (e.g. Gemma might require rotating OCR images, etc., but likely not).

MLX-VLM’s `processor` usually wraps a Hugging Face `AutoProcessor` or similar under the hood, but for a new model you may not have that luxury. You can subclass an existing processor if similar. For example, if Gemma is similar to PaLI, you might use the same transforms as a PaLI model already in MLX-VLM.

Here’s a pseudocode for a custom processor:

```python
from PIL import Image
import numpy as np

class GemmaProcessor:
    def __init__(self, image_size=224):
        self.image_size = image_size
    def __call__(self, images):
        # images can be list of paths or PIL images
        processed = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img)
            # resize and center-crop if needed
            img = img.convert("RGB")
            img = img.resize((self.image_size, self.image_size))
            arr = np.array(img).astype("float32") / 255.0  # scale 0-1
            # Normalize (Imagenet mean/std)
            mean = np.array([0.485, 0.456, 0.406], dtype="float32")
            std = np.array([0.229, 0.224, 0.225], dtype="float32")
            arr = (arr - mean) / std
            # transpose HWC->CHW if needed by MLX (likely yes)
            arr = arr.transpose(2,0,1)
            processed.append(mx.tensor(arr))  # convert to MLX tensor
        return processed
```

This is a simple example. In practice, MLX-VLM’s built-in processor might handle this if you supply the right config. For instance, the `load_config` for a model often contains a `image_size` field or links to a `preprocessor_config.json`. If so, you can parse that and avoid hardcoding values.

**3. Define the Model Class (if needed):** The crux of adding a new model is implementing the model’s architecture using MLX. If the new model is very similar to an existing one, you might not need a brand new class – you could reuse one. For example, Kimi-VL is built on InternVL (a vision model) plus a 1B param language model; if MLX-VLM had an InternVL-based class or if it’s similar to CLIP-ViT + small LLM, you could adapt the existing code. However, if it’s a novel architecture, you create a class.

In MLX-VLM’s codebase, there is likely a module (perhaps `mlx_vlm/models.py` or a folder) where different model architectures are defined. For instance, a `QwenVLModel` class, a `LLaVAModel` class, etc. You would add a `GemmaModel` class. This class should inherit from a common base (maybe `VLMModel` or directly use `mx.Model` from MLX). It needs to define how to **forward propagate** with image and text inputs. Typically:

* Initialize sub-components: e.g. an image encoder (vision transformer) and a text decoder (transformer decoder).
* In `forward` or `generate` method: run the image encoder on image tensor to get image embeddings; then feed those embeddings and text inputs to the language model.

Pseudocode outline for Gemma (just as an example):

```python
import mlx
class GemmaModel:
    def __init__(self, model_weights_dir):
        # Load or define layers
        # e.g., Vision Transformer backbone
        self.vision = mx.VisionTransformer(..., weights_file=model_weights_dir/"vision.safetensors")
        # Language model (transformer decoder)
        self.text = mx.AutoRegressiveTransformer(..., weights=model_weights_dir/"language.safetensors")
        # Perhaps a projection layer if needed to map vision outputs into text embedding space
        self.vision_proj = mx.Linear(...)

    def generate(self, input_ids, image_tensors, **kwargs):
        # 1. Encode images
        img_feats = None
        if image_tensors:
            img_feat_seq = self.vision(image_tensors)       # e.g., produces [batch, img_tokens, dim]
            img_feats = self.vision_proj(img_feat_seq)      # align to text embedding dimension
        # 2. Run language model with image features as prefix if applicable
        output_tokens = self.text.generate(input_ids, visual_prefix=img_feats, **kwargs)
        return output_tokens
```

The above is highly schematic. In practice, you need to match how the original model integrates the image. Some models prepend a special token that signals an image and the model learns to attend to an image embedding that’s appended at that token’s position. Others (like Flamingo/IDEFICS) use a gated cross-attention mechanism – that could be more complex to implement (involving modifications to the attention layers). If that’s the case, you might need to implement custom attention in MLX (which might require stepping into lower-level MLX ops).

For simpler cases, e.g., Kimi-VL (InternVL 1B) – that likely uses an InternImage or ViT and a simple language head – you can possibly adapt an existing ViT encoder class and a small decoder. For Qwen’s QVQ-72B, since QVQ is a variant of Qwen, its architecture (72B transformer with vision embedding) might reuse the Qwen2.5-VL implementation with just different weights. In that scenario, **no new code** is needed; once the weights are converted, you could try `load("mlx-community/QVQ-72B-Preview-?...")` and it might just work if the config is recognized as Qwen. If it’s not, you might trick MLX-VLM by using the Qwen loader and passing the QVQ weights (if they share architecture). The safer approach is to add a new entry for QVQ in the code, perhaps aliasing it to the Qwen class.

**4. Register the Model in MLX-VLM’s Loader:** After writing the new model class, you have to ensure the `load()` function knows about it. There may be a factory or mapping from model names or architecture names to the class. For example, MLX-VLM might inspect `config.json` from the model repo to see `"architectures": ["InternVLForConditionalGeneration"]` or a model type field. You can handle this by checking the model\_path or config:

```python
def load(model_path):
    # ... (download files)
    arch = config.get("architectures", [""])[0].lower()
    if "internvl" in arch or "kimi" in model_path.lower():
        model = GemmaModel(weights_dir)
        processor = GemmaProcessor()
    elif "gemma" in arch:
        model = GemmaModel(weights_dir)
        processor = GemmaProcessor()
    elif "qvq" in model_path.lower():
        model = QwenVLModel(weights_dir)  # reuse Qwen
        processor = QwenProcessor()       # reuse Qwen’s processor
    # ... other cases
    return model, processor
```

(This is illustrative – the actual `load` likely is more sophisticated.)

Essentially, you add conditions so that your new `GemmaModel` is used when appropriate. Also ensure the tokenizer is loaded (if the text vocab differs, load it via `mlx_lm` or Hugging Face tokenizers). For Gemma, if it uses a standard tokenizer (maybe T5 or similar, not sure), you might call `processor.tokenizer = AutoTokenizer.from_pretrained(original_model_name)` with `trust_remote_code` if needed. MLX-VLM often delegates tokenizer loading to Hugging Face’s Transformers library.

**5. Testing and Iterating:** With the above in place, you’d reinstall your modified `mlx_vlm` (or run it in a dev environment) and attempt to load and generate with the new model. You might start with a small input (maybe an empty image or a simple prompt) to see if everything wires up. If there are errors in shapes or missing implementations, you’d debug those (common issues are mismatched hidden dimensions between the vision and text model, or forgetting to add special tokens to the tokenizer). Once it runs, compare output with expected results (if the original model card has examples, see if MLX-VLM produces similar answers).

**6. Contribute back (optional):** Ideally, if you have successfully integrated a new model, consider contributing it to the MLX-VLM project via a pull request so others can use it. The maintainers (Prince Canuma and team) are actively adding new models, so they might have it in progress already – communication helps avoid duplicate work.

**Example: Adding Kimi-VL-A3B** – Kimi-VL is based on InternVL (a vision model from OpenGVLab) + a language model. Suppose MLX-VLM didn’t support InternVL yet. You would implement InternVL’s backbone (which might involve convolutions and region features; InternVL is a bit different from pure ViT). However, since Kimi’s base is only 1B, a workaround could be to treat it similarly to a smaller ViT. You might try using an existing ViT encoder class with the same image resolution (InternVL uses a 448px input if I recall correctly) and see if results are reasonable. If not, a custom implementation of InternImage architecture in MLX would be needed (which is quite advanced – perhaps why Kimi is not mainstream). Once done, you integrate as above. Luckily, in our scenario, Kimi’s MLX conversion exists, so likely the library got updated to handle it (maybe by treating it as a Qwen-style image token approach).

**Example: Qwen QVQ-72B** – Since QVQ is essentially Qwen 2.5 with an extended thinking process, integration might just require adding the model weights. The conversion might bring it in as a new model name. You could load it with the existing Qwen2.5 class if the tokenizer and architecture are identical, by passing the right `eos_token` if needed (Qwen needed that in MLX-LM). If the library refuses (unknown model), just add an alias: when model name contains “QVQ”, use Qwen’s implementation. This kind of light wiring often suffices for models that are variations of a supported family.

**Example: Gemma-3 27B** – This model is a flavor of PaLI/GEMMA. If MLX-VLM already has PaLI-X (which is Google’s earlier vision-language), you might extend that. If not, you create as above. Gemma might use a T5 text backbone (common in Google’s vision models) – meaning the text model is an encoder-decoder, not just decoder. MLX-VLM so far dealt with decoder-only causal models (since most chatty models are like that). Supporting an encoder-decoder (if needed for Gemma) is a bigger change – you’d need to run the text encoder on the prompt (maybe the prompt is just the image context in some tasks). This is an advanced scenario, but doable by leveraging MLX’s sequence-to-sequence capabilities. You might postpone that by focusing on the *instruct* fine-tune of Gemma which possibly converted it to a decoder-only format (some multi-modal instruct models wrap an encoder-decoder into a single stage).

**Summary of Steps:** Download/convert the model, implement processing and model forward logic, register it in load, then test. While adding a new model requires careful work, the *infrastructure* provided by MLX-VLM and MLX means you don’t start from scratch – you use highly optimized components and just arrange them to mirror the reference architecture. Always double-check memory requirements; new models might push the limits of even high-end Macs.

In many cases, you might not have to do any of this because the **MLX Community will have added support by the time you need it** – for example, a few months after a model release, a new MLX-VLM version might include it. Indeed, the MLX team often updates the library’s support in tandem with model conversions (noting that sometimes the conversion is done first and the official support comes slightly later). But if you’re eager or have a custom model, these steps allow you to extend MLX-VLM on your own.

---

**References:**

* MLX-VLM README and usage examples on GitHub.
* Discussion of MLX vs PyTorch on Hacker News (MLX is Apple’s array framework for GPUs).
* Strathweb’s blog on MLX (iOS) – notes that only converted models from `mlx-community` can be used.
* DZone Tutorial on Vision AI with MLX-VLM, which covers the efficiency of MLX’s unified memory for vision models.
* MLX Community Hugging Face page and model cards for specific converted models (e.g. Kimi-VL, Qwen2.5-72B).
* Simon Willison’s notes on running vision models with MLX-VLM.
* MLX-LM documentation for reference on model support and usage, which parallels MLX-VLM for text models.
