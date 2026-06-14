from html import escape

import gradio as gr

from database import get_all_recipes
from pipeline import run_pipeline


PAGE_SIZE = 6


def format_numbered_list(items):
    if not items:
        return ""

    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


def render_thumbnail_html(thumbnail_url):
    if not thumbnail_url:
        return ""

    escaped_url = escape(thumbnail_url, quote=True)
    return (
        f'<img src="{escaped_url}" '
        'style="width:100%; border-radius:8px; aspect-ratio:16/9; '
        'object-fit:cover;">'
    )


def add_recipe(youtube_url):
    try:
        recipe = run_pipeline(youtube_url)
    except ValueError as error:
        gr.Warning(str(error))
        return (
            "",
            "",
            "",
            "",
            gr.update(visible=False),
            gr.update(visible=False),
        )

    ingredients = format_numbered_list(recipe["ingredients"])
    steps = format_numbered_list(recipe["steps"])

    return (
        recipe["title"],
        render_thumbnail_html(recipe["thumbnail_url"]),
        ingredients,
        steps,
        gr.update(visible=False),
        gr.update(visible=True),
    )


def show_processing_status():
    return gr.update(
        value="Processing... this may take a minute while we fetch and analyse the transcript.",
        visible=True,
    )


def load_recipes(search_text):
    recipes = get_all_recipes()
    page = 1
    previous_update, next_update = get_pagination_button_updates(
        recipes,
        search_text,
        page,
    )
    return recipes, page, previous_update, next_update


def filter_recipes(recipes, search_text):
    search_text = (search_text or "").strip().lower()
    if not search_text:
        return recipes or []

    return [
        recipe
        for recipe in recipes or []
        if search_text in (recipe.get("title") or "").lower()
    ]


def get_total_pages(recipes):
    if not recipes:
        return 1

    return max(1, (len(recipes) + PAGE_SIZE - 1) // PAGE_SIZE)


def clamp_page(page, total_pages):
    return min(max(int(page or 1), 1), total_pages)


def get_pagination_button_updates(recipes, search_text, page):
    filtered_recipes = filter_recipes(recipes, search_text)
    total_pages = get_total_pages(filtered_recipes)
    page = clamp_page(page, total_pages)

    return (
        gr.update(interactive=page > 1),
        gr.update(interactive=page < total_pages),
    )


def reset_search_page(recipes, search_text):
    page = 1
    previous_update, next_update = get_pagination_button_updates(
        recipes,
        search_text,
        page,
    )
    return page, previous_update, next_update


def go_to_previous_page(recipes, search_text, current_page):
    filtered_recipes = filter_recipes(recipes, search_text)
    page = clamp_page((current_page or 1) - 1, get_total_pages(filtered_recipes))
    previous_update, next_update = get_pagination_button_updates(
        recipes,
        search_text,
        page,
    )
    return page, previous_update, next_update


def go_to_next_page(recipes, search_text, current_page):
    filtered_recipes = filter_recipes(recipes, search_text)
    page = clamp_page((current_page or 1) + 1, get_total_pages(filtered_recipes))
    previous_update, next_update = get_pagination_button_updates(
        recipes,
        search_text,
        page,
    )
    return page, previous_update, next_update


def render_recipe_cards(recipes, search_text, current_page):
    filtered_recipes = filter_recipes(recipes, search_text)
    total_pages = get_total_pages(filtered_recipes)
    current_page = clamp_page(current_page, total_pages)

    if not recipes:
        gr.Markdown("Load some recipes to display.")
        return

    if not filtered_recipes:
        gr.Markdown("No recipes match your search.")
        return

    start_index = (current_page - 1) * PAGE_SIZE
    page_recipes = filtered_recipes[start_index : start_index + PAGE_SIZE]

    gr.Markdown(
        f"Page {current_page} of {total_pages} - "
        f"{len(filtered_recipes)} recipe{'s' if len(filtered_recipes) != 1 else ''}"
    )

    for index, recipe in enumerate(page_recipes):
        title = recipe.get("title") or "Untitled recipe"
        thumbnail_url = recipe.get("thumbnail_url") or None
        url = recipe.get("url") or ""
        ingredients = format_numbered_list(recipe.get("ingredients") or [])
        steps = format_numbered_list(recipe.get("steps") or [])

        recipe_key = recipe.get("id") or f"{current_page}-{index}"
        with gr.Accordion(label=title, open=False, key=f"recipe-{recipe_key}"):
            if thumbnail_url:
                gr.HTML(render_thumbnail_html(thumbnail_url))
            if url:
                gr.Markdown(f"[Open original video]({url})")
            gr.Textbox(
                label="Ingredients",
                value=ingredients,
                lines=8,
                max_lines=8,
                elem_classes="scrollable-recipe-text",
            )
            gr.Textbox(
                label="Steps",
                value=steps,
                lines=8,
                max_lines=8,
                elem_classes="scrollable-recipe-text",
            )


css = """
.scrollable-recipe-text textarea {
    max-height: 220px;
    overflow-y: auto !important;
    resize: vertical;
}
"""


with gr.Blocks(theme='harsh8001/cartoon-style') as app:
    gr.Markdown("# The Recipe Exctractor")

    with gr.Tab("Add Recipe"):
        youtube_url_input = gr.Textbox(label="YouTube URL")
        submit_button = gr.Button("Submit", variant="primary")
        status_output = gr.Markdown(visible=False)

        with gr.Group(visible=False) as recipe_result_group:
            title_output = gr.Textbox(label="Recipe title")
            thumbnail_output = gr.HTML(label="Thumbnail")
            ingredients_output = gr.Textbox(label="Ingredients", lines=8)
            steps_output = gr.Textbox(label="Steps", lines=8)

        submit_button.click(
            fn=show_processing_status,
            inputs=None,
            outputs=status_output,
            show_progress="full",
        ).then(
            fn=add_recipe,
            inputs=youtube_url_input,
            outputs=[
                title_output,
                thumbnail_output,
                ingredients_output,
                steps_output,
                status_output,
                recipe_result_group,
            ],
            show_progress="full",
        )

    with gr.Tab("My Recipes"):
        search_input = gr.Textbox(
            label="Search recipes",
            placeholder="Search by title",
        )
        load_recipes_button = gr.Button("Load recipes")
        recipes_state = gr.State([])
        current_page_state = gr.State(1)

        @gr.render(inputs=[recipes_state, search_input, current_page_state])
        def render_saved_recipes(recipes, search_text, current_page):
            render_recipe_cards(recipes, search_text, current_page)

        with gr.Row():
            previous_button = gr.Button("Previous", interactive=False)
            next_button = gr.Button("Next", interactive=False)

        load_recipes_button.click(
            fn=load_recipes,
            inputs=search_input,
            outputs=[
                recipes_state,
                current_page_state,
                previous_button,
                next_button,
            ],
        )

        search_input.change(
            fn=reset_search_page,
            inputs=[recipes_state, search_input],
            outputs=[current_page_state, previous_button, next_button],
        )

        previous_button.click(
            fn=go_to_previous_page,
            inputs=[recipes_state, search_input, current_page_state],
            outputs=[current_page_state, previous_button, next_button],
        )

        next_button.click(
            fn=go_to_next_page,
            inputs=[recipes_state, search_input, current_page_state],
            outputs=[current_page_state, previous_button, next_button],
        )


app.launch(css=css)
