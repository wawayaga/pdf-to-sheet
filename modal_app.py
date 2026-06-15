import modal

modal_app = modal.App("tender-extractor")
app = modal_app

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
    gpu="A100",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def extract_tender_info(text):
    import os

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

    text = text[:100000]
    resumen_y_ficha = extract_resumen_y_ficha(generator, text)
    criterios_evaluacion = extract_criterios_evaluacion(generator, text)
    equipo_profesional = extract_equipo_profesional(generator, text)
    objetivos = extract_objetivos(generator, text)

    return {
        "resumen": resumen_y_ficha.get("resumen", {}),
        "ficha": resumen_y_ficha.get("ficha", {}),
        "criterios_evaluacion": criterios_evaluacion,
        "equipo_profesional": equipo_profesional,
        "objetivos": objetivos,
    }


def extract_resumen_y_ficha(generator, text):
    prompt = (
        "<s>[INST] Eres un extractor de informacion para documentos de bases "
        "de licitacion. Extrae solo el resumen y la ficha de la licitacion. "
        "El mandante es la institución estatal que emitió el documento que estás analizando, "
        "por lo que es común encontrar el nombre del mandante en los primeros 1000 caracteres"
        " del texto como la institución (municipalidad, ministerio o seremi) que ofrece los fondos a concursar. "
        "El monto siempre es un número y está expresado en millones de pesos "
        "chilenos, o en decenas o centenas de UF. Frecuentemente aparece como 'monto maximo disponible'. "
        "Es importante especificar el tipo de moneda en el campo llamado monto. "
        "La fecha de cierre de la licitación se refiere a la fecha de cierre de "
        "ingreso de las Ofertas Técnicas en el sistema www.mercadopublico.cl. "
        "La duracion del contrato es el numero de meses que transcurren entre la fecha de la firma del contrato "
        "y el fin del contrato. "
        "Devuelve SOLO JSON valido, sin markdown, comentarios, explicaciones "
        "ni texto adicional. El JSON debe tener exactamente esta estructura:\n"
        "{\n"
        '  "resumen": {\n'
        '    "id_licitacion": "", "nombre": "", "descripcion": "",\n'
        '    "mandante": "", "fecha_cierre": "", "hora_cierre": "",\n'
        '    "monto": "", "duracion": ""\n'
        "  },\n"
        '  "ficha": {\n'
        '    "id_licitacion": "", "nombre": "", "mandante": "",\n'
        '    "fecha_cierre": "", "hora_cierre": "", "monto": "",\n'
        '    "duracion_contrato": ""\n'
        "  }\n"
        "}\n\n"
        'Si un campo no se encuentra, usa "No especificado". '
        "No inventes datos.\n\n"
        f"Texto del documento:\n{text} [/INST]"
    )
    return run_json_prompt(generator, prompt)


def extract_criterios_evaluacion(generator, text):
    prompt = (
        "<s>[INST] Eres un extractor de informacion para documentos de bases "
        "de licitacion. Extrae solo los criterios de evaluacion. "
        "Los criterios de evaluación siempre se encuentran en una tabla "
        "que contiene las palabras 'Criterios de Evaluación' o 'Criterios de Evaluacion', "
        "y esta contiene al menos dos columnas, una para el nombre del criterio y otra para la ponderación del puntaje total. "
        "La ponderación del puntaje total puede aparecer como porcentaje o como fracción. "
        "Devuelve SOLO JSON valido, sin markdown, comentarios, explicaciones "
        "ni texto adicional. El JSON debe tener exactamente esta estructura:\n"
        "{\n"
        '  "criterios_evaluacion": [\n'
        '    {"criterio": "", "descripcion": "", "ponderacion": "", '
        '"numero_anexo": ""}\n'
        "  ]\n"
        "}\n\n"
        "criterios_evaluacion debe contener un objeto por cada item encontrado "
        "y puede ser una lista vacia si no hay items. "
        'Si un campo no se encuentra, usa "No especificado". '
        "No inventes datos.\n\n"
        f"Texto del documento:\n{text} [/INST]"
    )
    data = run_json_prompt(generator, prompt)
    return data.get("criterios_evaluacion", [])


def extract_equipo_profesional(generator, text):
    prompt = (
        "<s>[INST] Eres un extractor de informacion para documentos de bases "
        "de licitacion. Extrae solo el equipo profesional requerido. "
        "El equipo profesional es un listado roles que son necesarios "
        "para la ejecución del servicio. Estos profesionales son solicitados expresamente"
        " y de forma explícita en el texto. No es algo que tienes que suponer, sino que encontrar de forma textual"
        "Los profesionales requeridos pueden tener requisitos como licenciaturas, máster, "
        "doctorado, diplomados, o especialidad en alguna área. Estos requisitos deben "
        "estar especificados en el key 'requisitos' dentro del JSON. "
        "Devuelve SOLO JSON valido, sin markdown, comentarios, explicaciones "
        "ni texto adicional. El JSON debe tener exactamente esta estructura:\n"
        "{\n"
        '  "equipo_profesional": [\n'
        '    {"rol": "", "requisitos": ""}\n'
        "  ]\n"
        "}\n\n"
        "equipo_profesional debe contener un objeto por cada item encontrado "
        "y puede ser una lista vacia si no hay items. "
        'Si un campo no se encuentra, usa "No especificado". '
        "No inventes datos.\n\n"
        f"Texto del documento:\n{text} [/INST]"
    )
    data = run_json_prompt(generator, prompt)
    return data.get("equipo_profesional", [])


def extract_objetivos(generator, text):
    prompt = (
        "<s>[INST] Eres un extractor de informacion para documentos de bases "
        "de licitacion. Extrae solo el objetivo general y los objetivos "
        "especificos del servicio o proyecto licitado. "
        "El objetivo general suele describir el proposito principal de la "
        "contratacion. Los objetivos especificos suelen aparecer como una "
        "lista, numeracion o viñetas que detallan resultados, actividades o "
        "alcances concretos. "
        "Devuelve SOLO JSON valido, sin markdown, comentarios, explicaciones "
        "ni texto adicional. El JSON debe tener exactamente esta estructura:\n"
        "{\n"
        '  "objetivos": {\n'
        '    "objetivo_general": "",\n'
        '    "objetivos_especificos": ""\n'
        "  }\n"
        "}\n\n"
        "Si hay varios objetivos especificos, agrupalos en un solo texto "
        "separado por punto y coma. "
        'Si un campo no se encuentra, usa "No especificado". '
        "No inventes datos.\n\n"
        f"Texto del documento:\n{text} [/INST]"
    )
    data = run_json_prompt(generator, prompt)
    return data.get("objetivos", {})


def run_json_prompt(generator, prompt, max_new_tokens=2048):
    import json

    def generate_text(text_prompt):
        return generator(
            text_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0,
            return_full_text=False,
        )[0]["generated_text"].strip()

    def extract_json_object(output):
        start = output.find("{")
        if start == -1:
            raise ValueError(f"Model did not return a JSON object: {output}")

        depth = 0
        in_string = False
        escaped = False
        for index, character in enumerate(output[start:], start=start):
            if escaped:
                escaped = False
                continue
            if character == "\\":
                escaped = True
                continue
            if character == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return output[start : index + 1]

        raise ValueError(f"Model returned an incomplete JSON object: {output}")

    def parse_json_output(output):
        json_text = extract_json_object(output)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Model returned invalid JSON "
                f"at line {error.lineno}, column {error.colno}: {error.msg}. "
                f"Output starts with: {output[:1000]}"
            ) from error

    output = generate_text(prompt)
    try:
        return parse_json_output(output)
    except ValueError:
        repair_prompt = (
            "<s>[INST] Corrige el siguiente texto para que sea SOLO JSON "
            "valido. No agregues markdown, comentarios ni explicaciones. "
            "No cambies los nombres de las claves ni inventes datos. "
            "Devuelve solo el objeto JSON corregido.\n\n"
            f"Texto a corregir:\n{output} [/INST]"
        )
        repaired_output = generate_text(repair_prompt)
        return parse_json_output(repaired_output)
