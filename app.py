import gradio as gr

from pipeline import run_pipeline


def get_uploaded_pdf_path(uploaded_pdf):
    if uploaded_pdf is None:
        raise ValueError("Sube un archivo PDF para procesar.")

    if isinstance(uploaded_pdf, str):
        return uploaded_pdf

    return uploaded_pdf.name


def process_pdf(uploaded_pdf):
    try:
        pdf_path = get_uploaded_pdf_path(uploaded_pdf)
        extracted_info, excel_path = run_pipeline(pdf_path)
    except ValueError as error:
        gr.Warning(str(error))
        return (
            None,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    return (
        extracted_info,
        excel_path,
        gr.update(visible=False),
        gr.update(visible=True),
    )


def show_processing_status():
    return gr.update(
        value="Procesando... esto puede tardar unos minutos",
        visible=True,
    )


with gr.Blocks(theme="harsh8001/cartoon-style") as app:
    gr.Markdown("# Extractor de Licitaciones")

    with gr.Tab("Extractor"):
        pdf_input = gr.File(label="Subir PDF", file_types=[".pdf"])
        submit_button = gr.Button("Procesar", variant="primary")
        status_output = gr.Markdown(visible=False)

        with gr.Group(visible=False) as result_group:
            extracted_output = gr.JSON(label="Información extraída")
            excel_output = gr.File(label="Descargar Excel")

        submit_button.click(
            fn=show_processing_status,
            inputs=None,
            outputs=status_output,
            show_progress="full",
        ).then(
            fn=process_pdf,
            inputs=pdf_input,
            outputs=[
                extracted_output,
                excel_output,
                status_output,
                result_group,
            ],
            show_progress="full",
        )


if __name__ == "__main__":
    app.launch()
