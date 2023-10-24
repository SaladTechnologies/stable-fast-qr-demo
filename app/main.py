from model import load, warmup_config, get_qr_control_image
import os
import csv
import time

start = time.perf_counter()
pipe = load()
print(f"Total startup time: {time.perf_counter() - start}s", flush=True)

url_dir = os.getenv("URL_DIR", "/urls")
output_dir = os.getenv("OUTPUT_DIR", "/output")


def all_urls_to_run():
    # Loop through all files in the directory, sorted by filename
    for filename in sorted(os.listdir(url_dir)):
        if filename.endswith(".csv"):
            with open(os.path.join(url_dir, filename), newline="") as csvfile:
                reader = csv.reader(csvfile, delimiter=",", quotechar="|")
                for row in reader:
                    url, prompt, output_filename = row
                    yield url, prompt, output_filename


if __name__ == "__main__":
    for url, prompt, output_filename in all_urls_to_run():
        print(f"Running {url} with prompt {prompt} and output file {output_filename}")
        start = time.perf_counter()
        qr = get_qr_control_image(url)
        qr_gen = time.perf_counter() - start
        config = warmup_config.copy()
        config["prompt"] = prompt
        config["image"] = qr
        image = pipe(**config).images[0]
        image_gen = time.perf_counter() - start - qr_gen
        image.save(os.path.join(output_dir, output_filename))
        total_time = time.perf_counter() - start
        print(f"Total time: {total_time}")
        print(f"Base QR generation time: {qr_gen}")
        print(f"Image generation time: {image_gen}")
        print(f"Image saving time: {total_time - image_gen - qr_gen}")
