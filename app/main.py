import time

start = time.perf_counter()

import torch
from model import load, get_qr_control_image
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import io


pipe = load()
print(f"Total startup time: {time.perf_counter() - start}s", flush=True)

host = os.getenv("HOST", "0.0.0.0")
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
    num_inference_steps: int = 50
    controlnet_conditioning_scale: float = 1.0
    guidance_scale: float = 7.5
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0
    seed: Optional[int] = None


class GenerateRequest(BaseModel):
    url: str
    params: GenerateParams


@app.post("/generate")
async def generate(request: GenerateRequest):
    start = time.perf_counter()
    url = request.url
    params = request.params
    qr = get_qr_control_image(url)
    qr_gen = time.perf_counter() - start
    config = params.dict()
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
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
