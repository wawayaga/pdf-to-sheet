import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY environment variables are required")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def save_recipe(title, ingredients, steps, url, thumbnail_url):
    recipe = {
        "title": title,
        "ingredients": ingredients,
        "steps": steps,
        "url": url,
        "thumbnail_url": thumbnail_url,
    }
    response = supabase.table("recipes").insert(recipe).execute()
    return response.data


def recipe_exists(url):
    response = supabase.table("recipes").select("id").eq("url", url).limit(1).execute()
    return len(response.data) > 0


def get_all_recipes():
    response = supabase.table("recipes").select("*").order("created_at", desc=True).execute()
    return response.data
