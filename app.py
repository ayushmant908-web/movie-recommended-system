
import os
import pickle
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

load_dotenv(
    os.path.join(BASE_DIR, ".env"),
    override=True
)

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR
)

print("TMDB API key loaded:", bool(TMDB_API_KEY))


# ==================================================
# 2. LOAD DATA
# ==================================================

movies = pd.read_pickle(
    os.path.join(BASE_DIR, "movie_list.pkl")
).reset_index(drop=True)

if "title" not in movies.columns:
    raise ValueError(
        "movie_list.pkl must contain a 'title' column."
    )

with open(
    os.path.join(BASE_DIR, "top5_simaliarity.pkl"), "rb"
) as file:
    top_movies = pickle.load(file)

if not isinstance(top_movies, dict):
    raise ValueError(
        "top5_simaliarity.pkl must contain a dictionary."
    )

print("Total movies:", len(movies))


# ==================================================
# 3. TMDB REQUEST HELPER
# ==================================================

def tmdb_get(url, params):
    """Make a TMDB request and return JSON or None."""

    try:
        response = requests.get(
            url,
            params=params,
            timeout=(3, 5)
        )

        if response.status_code != 200:
            print(
                "TMDB request failed:",
                response.status_code,
                response.text[:150]
            )
            return None

        return response.json()

    except (requests.RequestException, ValueError) as error:
        print("TMDB connection error:", error)
        return None


# ==================================================
# 4. FETCH POSTER
# ==================================================

@lru_cache(maxsize=4096)
def fetch_poster(movie_id, movie_title=""):

    if not TMDB_API_KEY:
        return None

    # Try the TMDB movie ID first.
    if movie_id is not None:
        try:
            if not pd.isna(movie_id):
                numeric_id = int(float(movie_id))

                data = tmdb_get(
                    f"https://api.themoviedb.org/3/movie/{numeric_id}",
                    {
                        "api_key": TMDB_API_KEY,
                        "language": "en-US"
                    }
                )

                if data:
                    poster_path = data.get("poster_path")

                    if poster_path:
                        return POSTER_BASE_URL + poster_path

        except (ValueError, TypeError, OverflowError):
            pass

    # If ID lookup fails, search by title.
    title = str(movie_title).strip()

    if not title:
        return None

    data = tmdb_get(
        "https://api.themoviedb.org/3/search/movie",
        {
            "api_key": TMDB_API_KEY,
            "query": title,
            "include_adult": "false",
            "language": "en-US",
            "page": 1
        }
    )

    if not data:
        return None

    results = data.get("results", [])

    # Prefer exact title matches that have posters.
    exact_matches = [
        item for item in results
        if item.get("title", "").strip().casefold()
        == title.casefold()
        and item.get("poster_path")
    ]

    if exact_matches:
        return POSTER_BASE_URL + exact_matches[0]["poster_path"]

    # Otherwise, use the first result that has a poster.
    for item in results:
        poster_path = item.get("poster_path")

        if poster_path:
            return POSTER_BASE_URL + poster_path

    return None


# ==================================================
# 5. PREPARE MOVIE
# ==================================================

def prepare_movie(row):

    movie = row.to_dict()

    movie_id = None

    for column in (
        "movie_id",
        "tmdb_id",
        "tmdbId",
        "movieId",
        "id"
    ):
        value = movie.get(column)

        if value is not None:
            try:
                if not pd.isna(value):
                    movie_id = value
                    break
            except (TypeError, ValueError):
                continue

    movie_title = str(movie.get("title", "")).strip()

    movie["poster_url"] = fetch_poster(
        movie_id,
        movie_title
    )

    return movie


def prepare_movies_parallel(dataframe):
    """Fetch posters concurrently instead of sequentially."""

    rows = [
        row for _, row in dataframe.iterrows()
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(
            executor.map(prepare_movie, rows)
        )


# ==================================================
# 6. COMMON TEMPLATE RENDERER
# ==================================================

def render_page(
    active_tab,
    home_movies=None,
    recommendations=None,
    selected_movie="",
    success=False,
    error=None
):
    return render_template(
        "index.html",
        active_tab=active_tab,
        home_movies=home_movies or [],
        movie_titles=movies["title"].dropna().astype(str).tolist(),
        recommendations=recommendations or [],
        selected_movie=selected_movie,
        success=success,
        error=error
    )


# ==================================================
# 7. HOME: FIRST 50 MOVIES
# ==================================================

@app.route("/")
def home():

    home_movies = prepare_movies_parallel(
        movies.head(50)
    )

    return render_page(
        active_tab="home",
        home_movies=home_movies
    )


# ==================================================
# 8. RECOMMENDATION: TOP 5
# ==================================================

@app.route("/recommend", methods=["GET", "POST"])
def recommendation():

    recommendations = []
    selected_movie = ""
    error = None

    if request.method == "POST":

        selected_movie = request.form.get(
            "movie", ""
        ).strip()

        matches = movies.index[
            movies["title"].astype(str) == selected_movie
        ].tolist()

        if not matches:
            error = "Movie not found. Please select another movie."

        else:
            movie_index = matches[0]

            indices = top_movies.get(
                movie_index,
                top_movies.get(str(movie_index), [])
            )

            if indices is None:
                indices = []

            valid_indices = []

            for item in indices:
                try:
                    # Supports plain indices and (index, score) pairs.
                    if isinstance(item, (list, tuple)):
                        idx = int(item[0])
                    else:
                        idx = int(item)

                    if (
                        0 <= idx < len(movies)
                        and idx != movie_index
                        and idx not in valid_indices
                    ):
                        valid_indices.append(idx)

                except (ValueError, TypeError, IndexError):
                    continue

                if len(valid_indices) == 5:
                    break

            if valid_indices:
                recommendations = prepare_movies_parallel(
                    movies.iloc[valid_indices]
                )
            else:
                error = "No recommendations found for this movie."

    return render_page(
        active_tab="recommend",
        recommendations=recommendations,
        selected_movie=selected_movie,
        error=error
    )


# ==================================================
# 9. CONTACT
# ==================================================

@app.route("/contact", methods=["GET", "POST"])
def contact():

    success = False
    error = None

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()

        if not name or not email or not message:
            error = "Please fill in all fields."

        elif (
            "@" not in email
            or "." not in email.split("@")[-1]
        ):
            error = "Please enter a valid email address."

        else:
            # Demo only; messages are not saved or emailed.
            success = True

    return render_page(
        active_tab="contact",
        success=success,
        error=error
    )


# ==================================================
# 10. RUN APPLICATION
# ==================================================

if __name__ == "__main__":
    app.run(debug=True)




