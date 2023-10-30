import re
import subprocess
import torch
import json
import os


num_vcpu = os.getenv("NUM_VCPU", "2")
mem_gb = os.getenv("MEM_GB", "12")

num_vcpu = int(num_vcpu)
mem_gb = int(mem_gb)

vcpu_cost = 0.004 * num_vcpu
mem_cost = 0.001 * mem_gb

with open("./gpu-cost.json") as f:
    gpu_cost = json.load(f)


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


def shorten_gpu_name(full_name):
    shortened = []
    for name in full_name.split("\n"):
        # Extract the GPU model number, any 'Ti' suffix, and "Laptop GPU" distinction
        match = re.search(r"(RTX|GTX) (\d{3,4})( Ti)?( Laptop GPU)?", name)
        if match:
            shortened.append(
                match.group(1)
                + " "
                + match.group(2)
                + (match.group(3) or "")
                + (" Laptop" if match.group(4) else "")
            )
        else:
            shortened.append(name)
    return " & ".join(shortened)


def get_gpu_info():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    vram = round(vram, 2)
    cost_per_hour = gpu_cost.get(shorten_gpu_name(gpu_name), 0.0) + vcpu_cost + mem_cost
    cost_per_second = cost_per_hour / 3600
    return gpu_name, vram, cost_per_second
