import time

start = time.perf_counter()

import torch
from model import load, get_qr_control_image, image_size
import os
from pathlib import Path
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, field_validator, ValidationInfo
from typing import Optional, Literal, Union
import json
import uvicorn
import io

gpu_name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
vram = round(vram, 2)
print(f"Using GPU {gpu_name} with {vram} gb VRAM", flush=True)

pipe = load()
# pipe = None
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


def parse_color_str(v: Union[str, tuple[int, int, int]]) -> tuple[int, int, int]:
    if isinstance(v, str):
        return tuple(int(x) for x in v.split(","))
    return v


class SolidFillParams(BaseModel):
    front_color: Union[str, tuple[int, int, int]] = (0, 0, 0)
    back_color: Union[str, tuple[int, int, int]] = (128, 128, 128)

    @field_validator("back_color", "front_color", mode="before")
    def parse_color(cls, v: Union[str, tuple[int, int, int]]) -> tuple[int, int, int]:
        return parse_color_str(v)


class RadialGradiantParams(BaseModel):
    center_color: Union[str, tuple[int, int, int]] = (0, 0, 0)
    back_color: Union[str, tuple[int, int, int]] = (128, 128, 128)
    edge_color: Union[str, tuple[int, int, int]] = (0, 0, 255)

    @field_validator("back_color", "center_color", "edge_color", mode="before")
    def parse_color(cls, v: Union[str, tuple[int, int, int]]) -> tuple[int, int, int]:
        return parse_color_str(v)


class SquareGradiantParams(BaseModel):
    center_color: Union[str, tuple[int, int, int]] = (0, 0, 0)
    back_color: Union[str, tuple[int, int, int]] = (128, 128, 128)
    edge_color: Union[str, tuple[int, int, int]] = (0, 0, 255)

    @field_validator("back_color", "center_color", "edge_color", mode="before")
    def parse_color(cls, v: Union[str, tuple[int, int, int]]) -> tuple[int, int, int]:
        return parse_color_str(v)


class HorizontalGradiantParams(BaseModel):
    left_color: Union[str, tuple[int, int, int]] = (0, 0, 0)
    back_color: Union[str, tuple[int, int, int]] = (128, 128, 128)
    right_color: Union[str, tuple[int, int, int]] = (0, 0, 255)

    @field_validator("back_color", "left_color", "right_color", mode="before")
    def parse_color(cls, v: Union[str, tuple[int, int, int]]) -> tuple[int, int, int]:
        return parse_color_str(v)


class VerticalGradiantParams(BaseModel):
    top_color: Union[str, tuple[int, int, int]] = (0, 0, 0)
    back_color: Union[str, tuple[int, int, int]] = (128, 128, 128)
    bottom_color: Union[str, tuple[int, int, int]] = (0, 0, 255)

    @field_validator("back_color", "top_color", "bottom_color", mode="before")
    def parse_color(cls, v: Union[str, tuple[int, int, int]]) -> tuple[int, int, int]:
        return parse_color_str(v)


class QRParams(BaseModel):
    error_correction: Literal["L", "M", "Q", "H"] = "M"
    drawer: Literal[
        "RoundedModule",
        "SquareModule",
        "GappedSquareModule",
        "CircleModule",
        "VerticalBars",
        "HorizontalBars",
    ] = "RoundedModule"
    color_mask: Optional[
        Literal[
            "SolidFill",
            "RadialGradiant",
            "SquareGradiant",
            "HorizontalGradiant",
            "VerticalGradiant",
        ]
    ] = "SolidFill"
    color_mask_params: Union[
        str,
        SolidFillParams,
        RadialGradiantParams,
        SquareGradiantParams,
        HorizontalGradiantParams,
        VerticalGradiantParams,
    ] = "ppppppp"

    @field_validator("color_mask_params", mode="before")
    @classmethod
    def parse_color_mask_params(
        cls,
        v: Union[
            str,
            SolidFillParams,
            RadialGradiantParams,
            SquareGradiantParams,
            HorizontalGradiantParams,
            VerticalGradiantParams,
        ],
        info: ValidationInfo,
    ) -> Union[
        SolidFillParams,
        RadialGradiantParams,
        SquareGradiantParams,
        HorizontalGradiantParams,
        VerticalGradiantParams,
    ]:
        if isinstance(v, str):
            v = json.loads(v)
        mask = info.data["color_mask"]
        if mask == "SolidFill":
            return SolidFillParams(**v)
        elif mask == "RadialGradiant":
            return RadialGradiantParams(**v)
        elif mask == "SquareGradiant":
            return SquareGradiantParams(**v)
        elif mask == "HorizontalGradiant":
            return HorizontalGradiantParams(**v)
        elif mask == "VerticalGradiant":
            return VerticalGradiantParams(**v)
        else:
            raise ValueError(f"Invalid color mask {mask}")


class GenerateRequest(BaseModel):
    url: str
    params: GenerateParams
    qr_params: QRParams


class PreviewRequest(QRParams):
    url: str


@app.post("/generate")
async def generate(request: Request):
    request = GenerateRequest(**await request.json())
    start = time.perf_counter()
    url = request.url
    params = request.params
    qr_params = request.qr_params.model_dump()
    qr = get_qr_control_image(url, size=image_size, **qr_params)
    qr_gen = time.perf_counter() - start
    config = params.model_dump()
    if "seed" in config and config["seed"] is not None:
        config["generator"] = torch.Generator("cuda").manual_seed(config["seed"])
    del config["seed"]
    config["image"] = qr
    config["width"] = image_size
    config["height"] = image_size
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
            "X-Total-VRAM": str(vram),
        },
    )


# use like get /qr?url=something. returns an image
@app.get("/qr")
async def get_qr(request: Request):
    # Get query string as a dict
    opts = PreviewRequest(**dict(request.query_params)).model_dump()
    qr = get_qr_control_image(**opts)
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
