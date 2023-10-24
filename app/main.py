import time

start = time.perf_counter()

import torch
from model import load, get_qr_control_image
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import io

gpu_name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"Using GPU {gpu_name} with {vram} VRAM", flush=True)

pipe = load()
print(f"Total startup time: {time.perf_counter() - start}s", flush=True)

host = os.getenv("HOST", "*")
port = os.getenv("PORT", "1234")

port = int(port)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/hc")
async def health_check():
    return "OK"


class GenerateParams(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: int = 50
    controlnet_conditioning_scale: float = 1.0
    guidance_scale: float = 7.5
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0
    seed: Optional[int] = None
    width: int = 512
    height: int = 512


class GenerateRequest(BaseModel):
    url: str
    params: GenerateParams


@app.post("/generate")
async def generate(request: GenerateRequest):
    start = time.perf_counter()
    url = request.url
    params = request.params
    qr = get_qr_control_image(url, size=params.width)
    qr_gen = time.perf_counter() - start
    config = params.model_dump()
    if "seed" in config and config["seed"] is not None:
        config["generator"] = torch.Generator("cuda").manual_seed(config["seed"])
    del config["seed"]
    config["image"] = qr
    image = pipe(**config).images[0]
    image_gen = time.perf_counter() - start - qr_gen
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="png")
    image_bytes.seek(0)
    total_time = time.perf_counter() - start

    return StreamingResponse(
        image_bytes,
        media_type="image/png",
        headers={
            "X-Total-Time": str(total_time),
            "X-QR-Generation-Time": str(qr_gen),
            "X-Image-Generation-Time": str(image_gen),
            "X-GPU-Name": gpu_name,
            "X-VRAM": str(vram),
        },
    )


# use like get /qr?url=something. returns an image
@app.get("/qr")
async def get_qr(url: str, size: int = 512):
    qr = get_qr_control_image(url, size=size)
    qr_bytes = io.BytesIO()
    qr.save(qr_bytes, format="png")
    qr_bytes.seek(0)
    return StreamingResponse(
        qr_bytes,
        media_type="image/png",
    )


html_path = Path(__file__).parent / "index.html"


@app.get("/")
def serve_index():
    return FileResponse(html_path, media_type="text/html")


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
