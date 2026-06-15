import gradio as gr
import traceback

from pipeline import run_pipeline


APP_THEME = "harsh8001/cartoon-style"

APP_CSS = """
.completion-audio {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    opacity: 0;
    pointer-events: none;
}
"""


def get_uploaded_pdf_path(uploaded_pdf):
    if uploaded_pdf is None:
        raise ValueError("Sube un archivo PDF para procesar.")

    if isinstance(uploaded_pdf, str):
        return uploaded_pdf

    return uploaded_pdf.name


def process_pdf(uploaded_pdf):
    try:
        pdf_path = get_uploaded_pdf_path(uploaded_pdf)
        _, excel_path = run_pipeline(pdf_path)
    except ValueError as error:
        gr.Warning(str(error))
        return (
            gr.update(value=None, visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "error.mp3",
        )
    except Exception as error:
        traceback.print_exc()
        gr.Error(f"El procesamiento falló: {error}")
        return (
            gr.update(value=None, visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "error.mp3",
        )

    return (
        gr.update(value=excel_path, visible=True),
        gr.update(visible=False),
        gr.update(visible=True),
        "success.wav",
    )


def show_processing_status():
    message = "Procesando... esto puede tardar unos minutos"
    return (
        gr.update(
            value=f'<span style="color: #333333;">{message}</span>',
            visible=True,
        ),
        gr.update(value=None),
    )


with gr.Blocks() as app:
    gr.Markdown("# Extractor de Licitaciones")

    with gr.Tab("Extractor"):
        pdf_input = gr.File(label="Subir PDF", file_types=[".pdf"])
        submit_button = gr.Button("Procesar", variant="primary")
        status_output = gr.Markdown(visible=False)
        completion_audio = gr.Audio(
            autoplay=True,
            show_label=False,
            buttons=[],
            editable=False,
            elem_classes=["completion-audio"],
        )

        with gr.Group(visible=False) as result_group:
            excel_output = gr.File(
                label="Descargar Excel",
                visible=False,
            )

        submit_button.click(
            fn=show_processing_status,
            inputs=None,
            outputs=[status_output, completion_audio],
            show_progress="full",
        ).then(
            fn=process_pdf,
            inputs=pdf_input,
            outputs=[
                excel_output,
                status_output,
                result_group,
                completion_audio,
            ],
            show_progress="full",
        )


if __name__ == "__main__":
    app.launch(theme=APP_THEME, css=APP_CSS)
