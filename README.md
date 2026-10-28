# Paiton

**Local AI, optimized for AMD Radeon.**

Run chat, coding, image and video models with Paiton's optimized GPU runtimes.
Language models integrate with vLLM; image and video packages provide APIs,
command-line tools or ComfyUI workflows.

[Start here](#start-here) · [Requirements](#requirements) · [Quick start](#quick-start) ·
[Models](#models) · [Documentation](#documentation) · [Support](#support)

## Start here

| I want to… | Model | Command (from the repository folder) | What I get |
| --- | --- | --- | --- |
| Chat and code fast on one R9700 | [Qwen3.8 27B](models/Qwen3.8-MXFP4-DFlash2/README.md), 3-bit | `bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh` | about 180 tok/s, 65K tokens per request |
| Coding agents and long documents on one R9700 | [Qwen3.8 27B](models/Qwen3.8-MXFP4-DFlash2/README.md), 3-bit | `bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --mode long-kv4` | 262K tokens per request, prefix caching, images with `--vision` |
| The most accurate answers on one R9700 | [Qwen3.8 27B](models/Qwen3.8-MXFP4-DFlash2/README.md), MXFP4 | `bash models/Qwen3.8-MXFP4-DFlash2/run-mxfp4.sh` | about 156 tok/s, 65K tokens per request |
| A larger model, fastest, on two R9700s | [Qwen3.8 Flash Next](models/Qwen3.8-Flash-Next/README.md), 3-bit | `bash models/Qwen3.8-Flash-Next/run-flashnext.sh --fn-worker native` | about 259 tok/s, 98K-token window; 128 GB+ of RAM |
| A larger model, most accurate, on two R9700s | [Qwen3.8 Flash Next](models/Qwen3.8-Flash-Next/README.md), 4-bit | `bash models/Qwen3.8-Flash-Next/run-flashnext-4bit.sh --fn-worker native` | the 16-bit model's task scores, about 245 tok/s; 196 GB+ of RAM |
| Coding agents on two R9700s | [Qwen3.8 Flash Next](models/Qwen3.8-Flash-Next/README.md), 4-bit | `bash models/Qwen3.8-Flash-Next/run-flashnext-4bit.sh --fn-worker native --prefix-caching` | 262K-token window; a repeated 200K-token prompt starts in under 1 s |
| Check my setup with a small model | [MiniCPM5-2B](models/MiniCPM5-2B/README.md) | [Quick start](#quick-start) | a small chat and coding model to test your setup |
| Create images or video | [Image models](#image-generation) · [Video models](#video-generation) | see the model guide | |
| Use a browser workspace | [Paiton Studio](https://github.com/Eliovp-BV/paiton-studio) | install on a Linux host | Paiton in your browser |
| Use my existing vLLM installation | [Advanced installation](#use-your-own-vllm-environment) | | |

Each model guide lists its downloads, host requirements and every other mode. The 27B 3-bit rows need one extra
download of 9.55 GB ([how](models/Qwen3.8-MXFP4-DFlash2/README.md#optional-the-3-bit-weights)).

### What's new (24 October)

- **Qwen3.8 Flash Next, new 3-bit weights.** GSM8K, HumanEval and MMLU-Pro scores with no significant difference to the
  16-bit model, at full speed from VRAM. The launcher downloads them automatically into a new folder. [3-bit page](models/Qwen3.8-Flash-Next/README-3bit.md)
- **Qwen3.8 Flash Next, prefix caching with speculative decoding** on the 4-bit (`--fn-worker native --prefix-caching`):
  262K-token window, repeated long prompts in under 1 s. [Flash Next guide](models/Qwen3.8-Flash-Next/README.md)
- **Qwen3.8 27B, 3-bit accuracy update.** Closer to the 16-bit model, same speed, task scores unchanged within noise; run
  the 3-bit download again. [27B guide](models/Qwen3.8-MXFP4-DFlash2/README.md)

## Requirements

| Requirement | What you need |
| --- | --- |
| **GPU** | Radeon AI PRO R9700, 32 GB. Most guides use one card; Flash Next requires two. Other GPUs have not been qualified. |
| **Host** | Linux with a working AMD GPU driver. Check the model guide for any required driver version. |
| **Containers** | Docker with access to `/dev/kfd` and `/dev/dri`. ComfyUI launchers also require Docker Compose. |
| **Memory and storage** | Check your model's guide before downloading. Requirements depend on the model and selected profile. |

The Docker launchers supply each model's inference environment. Keep the GPU
available to the selected model; run one model at a time on a single R9700.

<a id="run-with-a-managed-container"></a><a id="managed-containers"></a><a id="launch-commands"></a>

## Quick start

**Check your setup with MiniCPM5-2B**, a small model for chat and coding.
You need Git, Docker, `curl`, internet access for the first setup, and the GPU
requirements above. [Reserve 60 GB of disk space for the first installation](models/MiniCPM5-2B/BENCHMARKS.md),
including the container runtime.

### 1. Start the server

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
PAITON_BIND=127.0.0.1 ./models/MiniCPM5-2B/serve-docker.sh
```

The first launch downloads the container and model weights, then prepares the
runtime. Later launches reuse the cache. Leave this terminal running.

### 2. Send your first request

Open a second terminal. Check that the server is ready:

```bash
curl --fail -i http://127.0.0.1:8036/health
```

Once it returns **HTTP 200**, send a message:

```bash
curl --fail http://127.0.0.1:8036/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minicpm5-2b",
    "messages": [{"role": "user", "content": "Explain binary search in two sentences."}],
    "max_tokens": 256,
    "temperature": 0
  }'
```

The response contains the model's answer in `choices[0].message.content`.

**Connect a client:** API base URL `http://127.0.0.1:8036/v1` · model `minicpm5-2b`.
This example accepts connections from this machine only. The
[MiniCPM guide](models/MiniCPM5-2B/README.md) covers network access, API keys and terminal chat.

### 3. Stop or choose another model

```bash
docker stop paiton-minicpm5-2b
```

Your downloaded weights and caches are retained. To start again, run
`PAITON_BIND=127.0.0.1 ./models/MiniCPM5-2B/serve-docker.sh` from the repository
folder. For another model, follow its guide below.

<a id="model-library"></a><a id="reading-the-benchmarks"></a>

## Models

Each guide covers setup, profiles and support limits. Benchmark links include
the tested configuration, performance measurements and quality tradeoffs.

<a id="chat-reasoning-coding-and-visual-understanding"></a>

### Chat, reasoning and coding

These models provide an OpenAI-compatible API through vLLM.

| Model and guide | Main use | Benchmarks |
| --- | --- | --- |
| [Qwen3.8 27B](models/Qwen3.8-MXFP4-DFlash2/README.md) | Chat and coding on one R9700; accuracy, speed and long-conversation profiles | [Results](models/Qwen3.8-MXFP4-DFlash2/README.md#benchmarks) |
| [Qwen3.8 Flash Next](models/Qwen3.8-Flash-Next/README.md) | Chat and coding on two R9700s; 3-bit or 4-bit, with optional image input on supported 4-bit profiles | [Results](models/Qwen3.8-Flash-Next/BENCHMARKS.md) |
| [MiniCPM5-2B](models/MiniCPM5-2B/README.md) | Lightweight chat, coding and tools | [Results](models/MiniCPM5-2B/BENCHMARKS.md) |
| [Qwen3.8 NEO CODER MAX 27B](models/Qwen3.8-NEO-CODER-MAX/README.md) | Coding and visual chat | [Results](models/Qwen3.8-NEO-CODER-MAX/BENCHMARKS.md) |
| [Qwen3-Coder 30B A3B](models/Qwen3-Coder-30B/README.md) | Code generation, review, testing and tools | [Results](models/Qwen3-Coder-30B/BENCHMARKS.md) |
| [GPT-OSS-20B](models/GPT-OSS-20B/README.md) | Reasoning, coding, tools and structured outputs | [Results](models/GPT-OSS-20B/BENCHMARKS.md) |
| [Ornith 1.5 35B A3B](models/Ornith-1.5/README.md) | Chat with optional reasoning | [Results](models/Ornith-1.5/BENCHMARKS.md) |

Earlier release: [Qwen3.8 27B Qronos](models/Qwen3.8/README.md), superseded by the
current Qwen3.8 27B guide above.

### Image generation

| Model and guide | Main use | Benchmarks |
| --- | --- | --- |
| [FLUX.2 klein 4B](models/FLUX.2-klein/README.md) | Text to image through ComfyUI, web or CLI | [Results](models/FLUX.2-klein/BENCHMARKS.md) |
| [Qwen-Image-2.1](models/Qwen-Image-2.1/README.md) | Image generation and editing through an API or CLI | [Results](models/Qwen-Image-2.1/BENCHMARKS.md) |

### Video generation

| Model and guide | Main use | Benchmarks |
| --- | --- | --- |
| [FastWan FullAttn 5B](models/FastWan/README.md) | Text to silent video through ComfyUI | [Results](models/Wan2.2/BENCHMARKS.md) |
| [Wan2.2 TI2V-5B](models/Wan2.2/README.md) | Text or image to silent video through ComfyUI | [Results](models/Wan2.2/BENCHMARKS.md) |
| [MiniMax H3](models/MiniMax-H3/README.md) | Video with stereo audio through ComfyUI | [Results](models/MiniMax-H3/BENCHMARKS.md) |

<a id="meeting-recordings"></a><a id="meeting-notes"></a>

### Meeting notes — preview

[Meeting](models/Meeting/README.md) is a review candidate for transcripts,
speaker labels and timestamped notes from recordings. Review generated notes
against the recording. Live Teams capture and Studio integration are unsupported.
[Benchmarks and limits](models/Meeting/BENCHMARKS.md).

<a id="amd-instinct-mi350p"></a>

### AMD Instinct MI350P (CDNA4): public default

A one-command default for **Qwen3.8 Flash Next on one AMD Instinct MI350P**: stock vLLM 0.32 with AITER, serving
AMD's [Quark MXFP4 checkpoint](https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8). It is
unoptimized and contains nothing proprietary. The optimized Paiton engine for this GPU is proprietary and not part of
this repository; see https://eliovp.com/blog/paiton-qwen38-flash-next-amd-instinct-mi350p for its results.

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
PAITON_BIND=127.0.0.1 scripts/cdna4/launch-mi350p.sh
```

The first start downloads the model (about 125 GB) into `~/.cache/huggingface`. When the server is ready, it answers
on `http://127.0.0.1:8000/v1` with model name `qwen3.8-flash-next`, a 262,144-token context and up to 64 concurrent
requests. `scripts/cdna4/launch-mi350p.sh --help` lists the settings (port, context, request limit, local model
folder, API key). Our published MI350P numbers use a 450 W GPU power limit (default 600 W); the script does not
change it. To build the image yourself: `docker build -f Dockerfile.cdna4-default -t paiton-mi350p-default .`

<a id="run-in-your-vllm-environment"></a><a id="native-cli"></a><a id="native-serving-presets"></a><a id="existing-commands-remain-supported"></a>

## Use your own vLLM environment

Already have a compatible vLLM installation? The pip-installable plugin supports
named presets inside their **exact supported runtime environments**. It does not
install or replace vLLM, PyTorch or ROCm.

Follow the [native installation guide](docs/NATIVE_EXECUTION.md) for supported
combinations, installation and verification. Preset capabilities differ from the
container releases; use the model's container guide for its published release profiles.

<a id="where-to-find-things"></a><a id="model-weights-and-existing-downloads"></a>

## Documentation

| Looking for | Guide |
| --- | --- |
| Reuse downloaded weights, configure caches or prepare offline | [Model weights and caches](docs/MODEL_WEIGHTS.md) |
| Install into a supported vLLM environment | [Native installation](docs/NATIVE_EXECUTION.md) |
| Use the older compatibility launcher | [Legacy vLLM launcher](docs/EXISTING_VLLM.md) |
| Build or validate native runtime packages | [Native packaging — contributors](docs/NATIVE_PACKAGING.md) |
| Download wheels, runtimes or model weights | [GitHub Releases](https://github.com/Eliovp-BV/paiton-vllm-plugin/releases) · [Hugging Face](https://huggingface.co/EliovpAI) |

## Support

For questions or problems, [open an issue](https://github.com/Eliovp-BV/paiton-vllm-plugin/issues).
Include your model and profile, GPU and system RAM, launch command, and relevant logs.

## About Paiton

This repository contains public integrations and compiled runtime artifacts.
The plugin is [Apache-2.0 licensed](LICENSE); the Paiton compiler is proprietary.
Model weights and bundled components retain their own licenses. See
[third-party notices](THIRD_PARTY_NOTICES.md) and each model's notices.

[Paiton by ElioVP](https://eliovp.com/products/paiton).
