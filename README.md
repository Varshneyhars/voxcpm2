# VoxCPM2 on RunPod Serverless

Production-ready serverless deployment of [OpenBMB VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) — a tokenizer-free, diffusion-autoregressive Text-to-Speech model (2B params, 30 languages, 48 kHz studio-quality output) — as a RunPod Serverless endpoint.

Call it like any HTTP API: send text (optionally with a reference voice clip), get back a base64-encoded WAV.

## Features

- **30-language TTS** — no language tag required
- **Voice Design** — generate a novel voice from a natural-language description
- **Controllable Cloning** — clone a voice from a short clip with optional style control
- **Ultimate Cloning** — reference audio + transcript for maximum fidelity
- **48 kHz output** via AudioVAE V2's built-in super-resolution
- Audio input via URL, base64, or `data:` URI
- Output as WAV or FLAC (base64 encoded)
- Robust bad-case retry with configurable thresholds

## Repository Layout

| File | Purpose |
|---|---|
| [handler.py](handler.py) | RunPod serverless entrypoint + `VoxCPM.generate` wrapper |
| [Dockerfile](Dockerfile) | PyTorch 2.5.1 + CUDA 12.1 image, optional weight prefetch |
| [requirements.txt](requirements.txt) | `runpod`, `voxcpm`, `soundfile`, `modelscope` |
| [test_input.json](test_input.json) | Sample payload for local testing |
| [client_example.py](client_example.py) | Python client for `/runsync` and `/run` endpoints |
| [.dockerignore](.dockerignore) / [.gitignore](.gitignore) | Build and VCS hygiene |

## Requirements

- **GPU**: NVIDIA with ≥ 8 GB VRAM (RTX 4090 / A5000 / L4 / A100 all work)
- **CUDA**: ≥ 12.0
- **Python**: ≥ 3.10, < 3.13 (handled by the Docker base image)
- **Disk**: ~15 GB container + ~6 GB for weights

## Quick Start

### 1. Build and push the image

```bash
docker build -t <your-registry>/voxcpm2-runpod:latest .
docker push <your-registry>/voxcpm2-runpod:latest
```

By default the weights are baked into the image (`PREFETCH_MODEL=true`) for fast cold starts. To skip baking and rely on a RunPod network volume instead:

```bash
docker build --build-arg PREFETCH_MODEL=false -t <your-registry>/voxcpm2-runpod:latest .
```

### 2. Create the RunPod endpoint

1. RunPod Console → **Serverless** → **New Endpoint**
2. Source: **Docker Image** → paste `<your-registry>/voxcpm2-runpod:latest`
3. GPU filter: RTX 4090 / A5000 / L4 / A100 (12 GB+ preferred)
4. Container disk: **20 GB**
5. *(Optional)* attach a **Network Volume** mounted at `/runpod-volume` so weights persist across workers when `PREFETCH_MODEL=false`
6. Max workers / idle timeout: tune to your traffic profile

### 3. Call the endpoint

```bash
export RUNPOD_API_KEY=rpa_xxx
export RUNPOD_ENDPOINT_ID=xxxxxxxx
python client_example.py
```

Or with `curl`:

```bash
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "Hello from VoxCPM2!", "cfg_value": 2.0, "inference_timesteps": 10}}'
```

## API

### Request (`input` object)

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | **required** | Text to synthesize. Prefix with `(description)` for Voice Design. |
| `reference_wav` | string | `null` | Reference clip for voice cloning. URL / base64 / `data:` URI. |
| `prompt_wav` | string | `null` | Prompt clip for Ultimate Cloning. URL / base64 / `data:` URI. |
| `prompt_text` | string | `null` | Exact transcript of `prompt_wav`. |
| `cfg_value` | float | `2.0` | LM guidance on LocDiT. Higher = stricter adherence. |
| `inference_timesteps` | int | `10` | LocDiT diffusion steps. Higher = better quality, slower. |
| `normalize` | bool | `true` | External text normalization. |
| `denoise` | bool | `true` | External denoiser on reference (requires `VOXCPM_LOAD_DENOISER=true`). |
| `retry_badcase` | bool | `true` | Retry unstable generations. |
| `retry_badcase_max_times` | int | `3` | Max retries. |
| `retry_badcase_ratio_threshold` | float | `6.0` | Length-ratio threshold for bad-case detection. |
| `output_format` | string | `"wav"` | `"wav"` or `"flac"`. |

### Response

```json
{
  "audio_base64": "UklGR...",
  "sample_rate": 48000,
  "format": "wav",
  "num_samples": 120000,
  "duration_seconds": 2.5
}
```

On failure:

```json
{ "error": "RuntimeError: <message>" }
```

## Usage Examples

### Basic TTS

```json
{"input": {"text": "VoxCPM2 brings multilingual support and voice cloning."}}
```

### Voice Design

```json
{"input": {"text": "(A young woman, gentle and sweet voice)Hello, welcome to VoxCPM2!"}}
```

### Controllable Cloning

```json
{"input": {
  "text": "This is a cloned voice.",
  "reference_wav": "https://example.com/speaker.wav"
}}
```

### Cloning with Style Control

```json
{"input": {
  "text": "(slightly faster, cheerful tone)This is a cloned voice with style control.",
  "reference_wav": "https://example.com/speaker.wav",
  "cfg_value": 2.0
}}
```

### Ultimate Cloning (highest fidelity)

```json
{"input": {
  "text": "This is an ultimate cloning demonstration.",
  "prompt_wav": "https://example.com/speaker.wav",
  "prompt_text": "The transcript of the reference audio.",
  "reference_wav": "https://example.com/speaker.wav"
}}
```

## Local Testing

Run the handler against `test_input.json` without RunPod:

```bash
pip install -r requirements.txt
python handler.py
```

The RunPod SDK will execute the local test file and print the result.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `VOXCPM_MODEL_ID` | `openbmb/VoxCPM2` | Hugging Face model ID |
| `VOXCPM_LOAD_DENOISER` | `false` | Load ZipEnhancer denoiser at startup (~2 GB extra) |
| `HF_HOME` / `HUGGINGFACE_HUB_CACHE` | `/opt/hf-cache` (prefetched) | Where weights live |

## Supported Languages

Arabic, Burmese, Chinese (+ dialects: 四川话, 粤语, 吴语, 东北话, 河南话, 陕西话, 山东话, 天津话, 闽南话), Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Tagalog, Thai, Turkish, Vietnamese.

## Performance

| Metric | Value |
|---|---|
| Parameters | 2 B |
| VRAM | ~8 GB |
| RTF (RTX 4090) | ~0.30 standard / ~0.13 with Nano-vLLM |
| Sample rate | 48 kHz output (16 kHz reference input) |
| dtype | bfloat16 |

## Cost Tips

- Bake weights into the image (`PREFETCH_MODEL=true`) if cold starts matter — no HF fetch on worker boot.
- Use a network volume (`PREFETCH_MODEL=false`) if you need smaller images and can tolerate the first-boot download.
- Set **idle timeout** to 5–10 s to avoid paying for idle workers between sporadic requests.
- Scale max workers conservatively — VoxCPM2 loads ~8 GB of weights per worker.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Use a larger GPU (≥ 12 GB) or set `VOXCPM_LOAD_DENOISER=false`. |
| Slow first request | Image didn't prefetch — rebuild with `PREFETCH_MODEL=true`, or keep a min-worker warm. |
| `denoise=true` fails | Set `VOXCPM_LOAD_DENOISER=true` in the endpoint env vars so ZipEnhancer loads. |
| Garbled or truncated output | Raise `retry_badcase_max_times`, tweak `retry_badcase_ratio_threshold`, or increase `inference_timesteps`. |
| Cloning quality low | Use Ultimate Cloning (pass `prompt_wav` + `prompt_text`), and use a clean 5–10 s reference clip. |

## License & Usage

- **Code (this repo)**: MIT
- **Model (VoxCPM2 weights)**: Apache-2.0 — commercial use permitted
- **Ethical use**: Impersonation, fraud, and disinformation are strictly forbidden. AI-generated speech should be clearly labeled.

## Citation

```bibtex
@article{voxcpm2_2026,
  title   = {VoxCPM2: Tokenizer-Free TTS for Multilingual Speech Generation, Creative Voice Design, and True-to-Life Cloning},
  author  = {VoxCPM Team},
  journal = {GitHub},
  year    = {2026}
}

@article{voxcpm2025,
  title   = {VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life Voice Cloning},
  author  = {Zhou, Yixuan and Zeng, Guoyang and Liu, Xin and Li, Xiang and
             Yu, Renjie and Wang, Ziyang and Ye, Runchuan and Sun, Weiyue and
             Gui, Jiancheng and Li, Kehan and Wu, Zhiyong and Liu, Zhiyuan},
  journal = {arXiv preprint arXiv:2509.24650},
  year    = {2025}
}
```

## Links

- Model: [huggingface.co/openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
- Upstream code: [github.com/OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
- Docs: [voxcpm.readthedocs.io](https://voxcpm.readthedocs.io/)
- RunPod Serverless: [docs.runpod.io/serverless](https://docs.runpod.io/serverless)
