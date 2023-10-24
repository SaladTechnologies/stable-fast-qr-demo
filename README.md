# stable-fast-qr-demo
A demo of using [stable-fast](https://github.com/chengzeyi/stable-fast) to generate qr codes

![](./qr.png)

## Prebuilt Docker Image

```
saladtechnologies/stable-fast-qr-demo:latest
```

## Docs

Swagger docs are available at `/docs` when the application is running.

## Build

```bash
./scripts/build
```

## Run

```bash
docker compose up
```

## Use

```bash
curl  -X POST \
  'http://localhost:1111/generate' \
  -H 'Content-Type: application/json' \
  --data '{
  "url": "https://github.com/SaladTechnologies/stable-fast-qr-demo",
  "params": {
    "prompt": "futuristic robot factory, gritty, realistic, rust, industrial",
    "guidance_scale": 3.5,
    "controlnet_conditioning_scale": 1.8,
    "control_guidance_start": 0.0,
    "control_guidance_end": 0.9,
    "num_inference_steps": 15
  }
}' > qr.png
```

## Configure

This application is configured with environment variables.

| Variable | Description | Default |
| --- | --- | --- |
| `PORT` | The port to listen on | `1234` |
| `HOST` | The host to listen on | `*` |
| `MODEL_DIR` | The directory to store models in | `./models` |
| `CIVITAI_CONTROLNET_MODEL` | A model version ID for a controlnet on Civit.ai | `122143` |
| `CIVITAI_CHECKPOINT_MODEL` | A model version ID for a checkpoint on Civit.ai | `128713` |