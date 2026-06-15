import tempfile
from pathlib import Path

import modal
import pandas as pd


MODAL_APP_NAME = "tender-extractor"

RESUMEN_COLUMNS = {
    "id_licitacion": "Id de la licitación",
    "nombre": "Nombre",
    "descripcion": "Descripción",
    "mandante": "Mandante",
    "fecha_cierre": "Fecha de cierre",
    "hora_cierre": "Hora de cierre",
    "monto": "Monto",
    "duracion": "Duración",
}

FICHA_COLUMNS = {
    "id_licitacion": "Id de la licitación",
    "nombre": "Nombre",
    "mandante": "Mandante",
    "fecha_cierre": "Fecha de cierre",
    "hora_cierre": "Hora de cierre",
    "monto": "Monto",
    "duracion_contrato": "Duración del contrato",
}

CRITERIOS_COLUMNS = {
    "criterio": "Criterio",
    "descripcion": "Descripción",
    "ponderacion": "Ponderación",
    "numero_anexo": "Número de anexo",
}

EQUIPO_COLUMNS = {
    "rol": "Rol",
    "requisitos": "Requisitos",
}


def get_modal_functions():
    extract_pdf_text = modal.Function.from_name(MODAL_APP_NAME, "extract_pdf_text")
    extract_tender_info = modal.Function.from_name(
        MODAL_APP_NAME,
        "extract_tender_info",
    )
    return extract_pdf_text, extract_tender_info


def build_dataframe(data, columns):
    rows = data if isinstance(data, list) else [data or {}]
    return pd.DataFrame(rows).reindex(columns=columns.keys()).rename(columns=columns)


def save_excel(tender_info):
    output_file = tempfile.NamedTemporaryFile(
        suffix=".xlsx",
        prefix="licitacion_",
        delete=False,
    )
    output_path = output_file.name
    output_file.close()

    sheets = {
        "Resumen de la licitación": build_dataframe(
            tender_info.get("resumen"),
            RESUMEN_COLUMNS,
        ),
        "Ficha de la licitación": build_dataframe(
            tender_info.get("ficha"),
            FICHA_COLUMNS,
        ),
        "Criterios de evaluación": build_dataframe(
            tender_info.get("criterios_evaluacion", []),
            CRITERIOS_COLUMNS,
        ),
        "Equipo profesional": build_dataframe(
            tender_info.get("equipo_profesional", []),
            EQUIPO_COLUMNS,
        ),
    }

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, dataframe in sheets.items():
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)

    return output_path


def run_pipeline(pdf_path):
    pdf_bytes = Path(pdf_path).read_bytes()

    with modal.enable_output():
        extract_pdf_text, extract_tender_info = get_modal_functions()
        text = extract_pdf_text.remote(pdf_bytes)
        tender_info = extract_tender_info.remote(text)

    excel_path = save_excel(tender_info)
    return tender_info, excel_path
