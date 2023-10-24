import torch
import os
import time
import xformers
import triton
import requests
import subprocess
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    __version__ as diffusers_version,
)
from transformers import CLIPImageProcessor, __version__ as transformers_version
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from sfast.compilers.stable_diffusion_pipeline_compiler import (
    compile,
    CompilationConfig,
)
import qrcode

torch.backends.cuda.matmul.allow_tf32 = True

print("Torch version:", torch.__version__, flush=True)
print("XFormers version:", xformers.__version__, flush=True)
print("Triton version:", triton.__version__, flush=True)
print("Diffusers version:", diffusers_version, flush=True)
print("Transformers version:", transformers_version, flush=True)
print("CUDA Available:", torch.cuda.is_available(), flush=True)

compile_config = CompilationConfig.Default()
compile_config.enable_xformers = True
compile_config.enable_triton = True
compile_config.enable_cuda_graph = False
compile_config.memory_format = torch.channels_last
compile_config.enable_cuda_graph = True


civitai_base_url = "https://civitai.com/api/v1/model-versions/"

safety_checker_model = os.getenv(
    "HF_SAFETY_CHECKER", "CompVis/stable-diffusion-safety-checker"
)
feature_extractor_model = os.getenv("HF_CLIP_MODEL", "openai/clip-vit-base-patch32")
model_dir = os.getenv("MODEL_DIR", "/models")
controlnet_dir = os.path.join(model_dir, "controlnet")
checkpoint_dir = os.path.join(model_dir, "checkpoints")
output_dir = os.getenv("OUTPUT_DIR", "/output")
civitai_controlnet_model = os.getenv("CIVITAI_CONTROLNET_MODEL", None)
civitai_checkpoint_model = os.getenv("CIVITAI_CHECKPOINT_MODEL", None)

if not civitai_controlnet_model:
    raise ValueError("CIVITAI_CONTROLNET_MODEL is not set")

if not civitai_checkpoint_model:
    raise ValueError("CIVITAI_CHECKPOINT_MODEL is not set")


# Create the directories
os.makedirs(model_dir, exist_ok=True)
os.makedirs(controlnet_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)


def get_git_repo_url():
    try:
        url = (
            subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
            .decode("utf-8")
            .strip()
        )
        return url
    except subprocess.CalledProcessError:
        print("Error fetching repository URL. Are you inside a Git repository?")
        return "https://salad.com/"


def get_qr_control_image(url):
    qr = qrcode.QRCode(
        version=7.4,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color=(0, 0, 0), back_color=(128, 128, 128))
    return img.resize((512, 512))


my_repository_url = get_git_repo_url()
print(f"Repository URL: {my_repository_url}", flush=True)
my_repo_qr = get_qr_control_image(my_repository_url)

warmup_config = {
    "prompt": "Painterly dreamscape, clouds, mystical",
    "width": 512,
    "height": 512,
    "image": my_repo_qr,
    "num_inference_steps": 15,
    "controlnet_conditioning_scale": 1.6,
    "guidance_scale": 5.0,
    "control_guidance_start": 0.5,
}


def load_safety_checker(load_only=False):
    print("Loading safety checker...", flush=True)
    start = time.perf_counter()
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        safety_checker_model, torch_dtype=torch.float16, cache_dir=model_dir
    )
    if not load_only:
        safety_checker.to("cuda")
    feature_extractor = CLIPImageProcessor.from_pretrained(
        feature_extractor_model, torch_dtype=torch.float16, cache_dir=model_dir
    )
    print(f"Loaded safety checker in {time.perf_counter() - start} seconds", flush=True)
    return safety_checker, feature_extractor


def download_file(url, filepath):
    # Use wget to download the file, with --content-disposition to get the filename
    cmd = f'wget -q "{url}" --content-disposition -O {filepath}'
    print(f"Running command {cmd}", flush=True)
    os.system(cmd)


def download_if_not_exists(url, filepath):
    # Check if file already exists and is not empty
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        print(f"File {filepath} already exists, skipping download", flush=True)
    else:
        print(f"Downloading file {filepath}...", flush=True)
        start = time.perf_counter()
        download_file(url, filepath)
        print(
            f"Downloaded file {filepath} in {time.perf_counter() - start}s seconds",
            flush=True,
        )


def get_model_and_config_from_civitai_payload(payload):
    model_file = None
    config_file = None
    for file in payload["files"]:
        if file["type"] == "Model":
            model_file = file
        elif file["type"] == "Config":
            config_file = file
    return model_file, config_file


def load_controlnet_model_from_civitai(model_version_id, load_only=False):
    model_info = requests.get(civitai_base_url + model_version_id).json()
    model_file, config_file = get_model_and_config_from_civitai_payload(model_info)
    model_name = model_info["model"]["name"]
    print(f"Downloading controlnet model {model_name}...", flush=True)
    if model_file:
        model_filename = model_file["name"]
        model_filepath = os.path.join(controlnet_dir, model_filename)
        download_if_not_exists(model_file["downloadUrl"], model_filepath)

    if config_file:
        config_filename = config_file["name"]
        config_filepath = os.path.join(controlnet_dir, config_filename)
        download_if_not_exists(config_file["downloadUrl"], config_filepath)

    print(f"Loading controlnet model {model_name}...", flush=True)
    start = time.perf_counter()
    model = ControlNetModel.from_single_file(model_filepath)
    if not load_only:
        model.to("cuda", memory_format=torch.channels_last)
    print(
        f"Loaded controlnet model {model_name} in {time.perf_counter() - start} seconds",
        flush=True,
    )
    return model


def download_checkpoint_from_civitai(model_version_id, load_only=False):
    model_info = requests.get(civitai_base_url + model_version_id).json()
    model_file, config_file = get_model_and_config_from_civitai_payload(model_info)
    model_name = model_info["model"]["name"]
    print(f"Loading checkpoint {model_name}...", flush=True)
    if model_file:
        model_filename = model_file["name"]
        model_filepath = os.path.join(checkpoint_dir, model_filename)
        download_if_not_exists(model_file["downloadUrl"], model_filepath)

    return model_filepath


def load_controlnet_pipeline(
    model_path, controlnet, feature_extractor=None, safety_checker=None
):
    print(f"Loading controlnet pipeline from {model_path}...", flush=True)
    start = time.perf_counter()
    pipeline = StableDiffusionControlNetPipeline.from_single_file(
        model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=checkpoint_dir,
        extract_ema=True,
        device_map="auto",
        load_safety_checker=False,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline.feature_extractor = feature_extractor
    pipeline.safety_checker = safety_checker
    print(
        f"Loaded controlnet pipeline in {time.perf_counter() - start} seconds",
        flush=True,
    )
    print(f"Compiling controlnet pipeline...", flush=True)
    start = time.perf_counter()
    pipeline.to("cuda")
    pipeline.text_encoder.eval()
    pipeline.unet.eval()
    pipeline.controlnet.eval()
    pipeline = compile(pipeline, compile_config)

    output_image = pipeline(**warmup_config).images[0]
    print(
        f"Compiled controlnet pipeline in {time.perf_counter() - start} seconds",
        flush=True,
    )
    with open(os.path.join(output_dir, "warmup.png"), "wb") as f:
        output_image.save(f, format="png")
        print(f"Wrote warmup image to {os.path.join(output_dir, 'warmup.png')}")

    return pipeline


def load():
    controlnet = load_controlnet_model_from_civitai(civitai_controlnet_model)
    checkpoint_path = download_checkpoint_from_civitai(civitai_checkpoint_model)
    safety_checker, feature_extractor = load_safety_checker()
    pipeline = load_controlnet_pipeline(
        checkpoint_path, controlnet, feature_extractor, safety_checker
    )
    return pipeline
