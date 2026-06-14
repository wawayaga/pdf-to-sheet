import modal

modal_app = modal.App("recipe-saver")

image = modal.Image.debian_slim().apt_install("ffmpeg").pip_install(
    "torch",
    "transformers",
    "huggingface_hub",
    "youtube-transcript-api",
)


@modal_app.function(image=image, timeout=1800)
def get_transcript(youtube_url):
    from urllib.parse import parse_qs, urlparse

    from youtube_transcript_api import YouTubeTranscriptApi

    parsed_url = urlparse(youtube_url)
    host = parsed_url.netloc.removeprefix("www.")

    if host == "youtu.be":
        video_id = parsed_url.path.strip("/").split("/")[0]
    elif parsed_url.path == "/watch":
        video_id = parse_qs(parsed_url.query).get("v", [""])[0]
    elif parsed_url.path.startswith(("/shorts/", "/embed/")):
        video_id = parsed_url.path.strip("/").split("/")[1]
    else:
        video_id = ""

    if not video_id:
        raise ValueError("Could not determine the YouTube video id from the URL.")

    transcript = YouTubeTranscriptApi().fetch(video_id)
    snippets = transcript.to_raw_data()
    return " ".join(snippet["text"] for snippet in snippets)


@modal_app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def extract_recipe(transcript):
    import json
    import os
    import re

    import torch
    from transformers import pipeline

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable is required")

    generator = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        token=hf_token,
        torch_dtype=torch.float16,
        device=0,
    )

    prompt = (
        "<s>[INST] Extract a recipe from the transcript below. "
        "Return only valid JSON with exactly these keys: "
        '"ingredients" (a list of strings) and "steps" (a list of strings). '
        "For every ingredient, always include the quantity and unit in the same "
        'string as the ingredient name when the transcript mentions one, such as '
        '"100ml water", "500g flour", or "2 tablespoons olive oil". '
        "Do not separate quantities, units, and ingredient names into different "
        "fields or strings. "
        "Do not include markdown, comments, or any extra text.\n\n"
        f"Transcript:\n{transcript} [/INST]"
    )
    output = generator(
        prompt,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0,
        return_full_text=False,
    )[0]["generated_text"].strip()

    json_match = re.search(r"\{.*\}", output, flags=re.DOTALL)
    if not json_match:
        raise ValueError(f"Model did not return a JSON object: {output}")

    recipe = json.loads(json_match.group(0))
    return {
        "ingredients": [str(item) for item in recipe.get("ingredients", [])],
        "steps": [str(item) for item in recipe.get("steps", [])],
    }
