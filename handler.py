"""RunPod serverless handler for VoxCPM2 TTS."""
import base64
import io
import os
import tempfile
from typing import Optional
from urllib.request import urlopen

import numpy as np
import runpod
import soundfile as sf
from voxcpm import VoxCPM

MODEL_ID = os.environ.get("VOXCPM_MODEL_ID", "openbmb/VoxCPM2")
LOAD_DENOISER = os.environ.get("VOXCPM_LOAD_DENOISER", "false").lower() == "true"

print(f"[init] Loading VoxCPM2 model: {MODEL_ID} (denoiser={LOAD_DENOISER})")
MODEL = VoxCPM.from_pretrained(MODEL_ID, load_denoiser=LOAD_DENOISER)
SAMPLE_RATE = MODEL.tts_model.sample_rate
print(f"[init] Model ready. Sample rate: {SAMPLE_RATE}")


def _materialize_audio(value: Optional[str]) -> Optional[str]:
    """Accept a URL, a data URI, or raw base64; return a local file path.

    Returns None when value is falsy. Caller is responsible for cleanup.
    """
    if not value:
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        if value.startswith(("http://", "https://")):
            with urlopen(value, timeout=60) as resp:
                tmp.write(resp.read())
        else:
            if value.startswith("data:"):
                value = value.split(",", 1)[1]
            tmp.write(base64.b64decode(value))
        tmp.flush()
        return tmp.name
    finally:
        tmp.close()


def _cleanup(*paths: Optional[str]) -> None:
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.unlink(p)
            except OSError:
                pass


def _encode_wav(wav: np.ndarray, sample_rate: int, fmt: str = "WAV") -> str:
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def handler(event):
    """RunPod serverless entry point.

    Input schema (event["input"]):
        text (str, required): Text to synthesize. Prefix with "(description)"
            to use voice design, e.g. "(A young woman, gentle)Hello".
        reference_wav (str, optional): URL / base64 / data-URI for voice cloning.
        prompt_wav (str, optional): URL / base64 / data-URI for ultimate cloning.
        prompt_text (str, optional): Transcript of prompt_wav.
        cfg_value (float, default 2.0): LM guidance on LocDiT.
        inference_timesteps (int, default 10): LocDiT diffusion steps.
        normalize (bool, default True): Enable external text normalization.
        denoise (bool, default True): Enable external denoiser on reference.
        retry_badcase (bool, default True): Retry on unstable generations.
        retry_badcase_max_times (int, default 3)
        retry_badcase_ratio_threshold (float, default 6.0)
        output_format (str, default "wav"): "wav" or "flac".

    Returns:
        {"audio_base64": "...", "sample_rate": int, "format": "wav"}
    """
    inp = event.get("input") or {}

    text = inp.get("text")
    if not text or not isinstance(text, str):
        return {"error": "Field 'text' is required and must be a string."}

    reference_wav_path = None
    prompt_wav_path = None
    try:
        reference_wav_path = _materialize_audio(inp.get("reference_wav"))
        prompt_wav_path = _materialize_audio(inp.get("prompt_wav"))

        wav = MODEL.generate(
            text=text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=inp.get("prompt_text"),
            reference_wav_path=reference_wav_path,
            cfg_value=float(inp.get("cfg_value", 2.0)),
            inference_timesteps=int(inp.get("inference_timesteps", 10)),
            normalize=bool(inp.get("normalize", True)),
            denoise=bool(inp.get("denoise", True)),
            retry_badcase=bool(inp.get("retry_badcase", True)),
            retry_badcase_max_times=int(inp.get("retry_badcase_max_times", 3)),
            retry_badcase_ratio_threshold=float(
                inp.get("retry_badcase_ratio_threshold", 6.0)
            ),
        )

        fmt = str(inp.get("output_format", "wav")).lower()
        sf_format = "FLAC" if fmt == "flac" else "WAV"
        audio_b64 = _encode_wav(wav, SAMPLE_RATE, fmt=sf_format)

        return {
            "audio_base64": audio_b64,
            "sample_rate": SAMPLE_RATE,
            "format": fmt,
            "num_samples": int(wav.shape[-1]),
            "duration_seconds": float(wav.shape[-1] / SAMPLE_RATE),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        _cleanup(reference_wav_path, prompt_wav_path)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
