import modal

modal_app = modal.App("tender-extractor")

image = (
    modal.Image.debian_slim()
    .apt_install("tesseract-ocr", "tesseract-ocr-spa", "poppler-utils")
    .pip_install(
        "pdfplumber",
        "pdf2image",
        "pytesseract",
        "pillow",
        "torch",
        "transformers",
        "huggingface_hub",
    )
)


@modal_app.function(image=image, timeout=1800)
def extract_pdf_text(pdf_bytes):
    import io

    import pdfplumber
    import pytesseract
    from pdf2image import convert_from_bytes

    page_texts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_texts.append(page.extract_text() or "")

    page_count = len(page_texts)
    extracted_text = "\n\n".join(text.strip() for text in page_texts).strip()
    average_chars_per_page = len(extracted_text) / page_count if page_count else 0

    if average_chars_per_page < 50:
        images = convert_from_bytes(pdf_bytes)
        page_texts = [
            pytesseract.image_to_string(image, lang="spa") for image in images
        ]

    return "\n\n".join(text.strip() for text in page_texts).strip()


@modal_app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def extract_tender_info(text):
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
        "<s>[INST] Eres un extractor de informacion para documentos de bases "
        "de licitacion. Extrae la informacion solicitada del texto entregado. "
        "Devuelve SOLO JSON valido, sin markdown, comentarios, explicaciones "
        "ni texto adicional. El JSON debe tener exactamente esta estructura:\n"
        "{\n"
        '  "resumen": {\n'
        '    "id_licitacion": "", "nombre": "", "descripcion": "",\n'
        '    "licitante": "", "fecha_cierre": "", "hora_cierre": "",\n'
        '    "monto": "", "duracion": ""\n'
        "  },\n"
        '  "ficha": {\n'
        '    "id_licitacion": "", "nombre": "", "licitante": "",\n'
        '    "fecha_cierre": "", "hora_cierre": "", "monto": "",\n'
        '    "duracion_contrato": ""\n'
        "  },\n"
        '  "criterios_evaluacion": [\n'
        '    {"criterio": "", "descripcion": "", "ponderacion": "", '
        '"numero_anexo": ""}\n'
        "  ],\n"
        '  "equipo_profesional": [\n'
        '    {"rol": "", "requisitos": ""}\n'
        "  ]\n"
        "}\n\n"
        'Si un campo no se encuentra, usa "No especificado". '
        "criterios_evaluacion y equipo_profesional deben contener un objeto "
        "por cada item encontrado, y pueden ser listas vacias si no hay items. "
        "No inventes datos.\n\n"
        f"Texto del documento:\n{text} [/INST]"
    )
    output = generator(
        prompt,
        max_new_tokens=2048,
        do_sample=False,
        temperature=0,
        return_full_text=False,
    )[0]["generated_text"].strip()

    json_match = re.search(r"\{.*\}", output, flags=re.DOTALL)
    if not json_match:
        raise ValueError(f"Model did not return a JSON object: {output}")

    return json.loads(json_match.group(0))
