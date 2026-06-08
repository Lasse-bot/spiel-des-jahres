import os
import json
import base64
import httpx
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GAMES_FILE = Path(__file__).parent / "games.json"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "")

with open(GAMES_FILE) as f:
    GAMES = json.load(f)


def get_conn():
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL nicht gesetzt")
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS scanned (
                    name TEXT PRIMARY KEY,
                    location TEXT,
                    rating INTEGER
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS extra_games (
                    name TEXT PRIMARY KEY,
                    year INTEGER,
                    location TEXT,
                    rating INTEGER
                )
            """)
        conn.commit()


@app.on_event("startup")
def startup():
    init_db()


def find_matching_game(detected_texts: list[str], game_list: list[dict]) -> str | None:
    detected = " ".join(detected_texts).lower()
    best_match = None
    best_score = 0
    for game in game_list:
        name = game["name"].lower()
        words = name.split()
        matches = sum(1 for word in words if len(word) > 3 and word in detected)
        score = matches / len(words) if words else 0
        if score > best_score and score >= 0.5:
            best_score = score
            best_match = game["name"]
    return best_match


async def call_vision_api(image_data: bytes) -> list[str]:
    encoded = base64.b64encode(image_data).decode("utf-8")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}",
            json={
                "requests": [{
                    "image": {"content": encoded},
                    "features": [{"type": "TEXT_DETECTION", "maxResults": 50}]
                }]
            },
            timeout=30,
        )
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Google Vision API Fehler ({response.status_code}): {response.text}")
    result = response.json()
    annotations = result.get("responses", [{}])[0].get("textAnnotations", [])
    return [a["description"] for a in annotations]


@app.get("/api/games")
def get_games():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name, location, rating FROM scanned")
            rows = {r["name"]: r for r in cur.fetchall()}
    return [
        {
            **game,
            "scanned": game["name"] in rows,
            "location": rows.get(game["name"], {}).get("location"),
            "rating": rows.get(game["name"], {}).get("rating"),
        }
        for game in GAMES
    ]


@app.post("/api/scan")
async def scan_image(
    file: UploadFile = File(...),
    location: Optional[str] = Form(None),
):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY nicht gesetzt")

    image_data = await file.read()
    texts = await call_vision_api(image_data)
    matched = find_matching_game(texts, GAMES)

    if not matched:
        return {"matched": False, "game": None, "detected_text": " ".join(texts[:5])}

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO scanned (name, location, rating)
                VALUES (%s, %s, NULL)
                ON CONFLICT (name) DO UPDATE SET location = EXCLUDED.location
            """, (matched, location or None))
        conn.commit()

    return {"matched": True, "game": matched}


@app.post("/api/rate/{game_name}")
def rate_game(game_name: str, rating: Optional[int] = Form(None), location: Optional[str] = Form(None)):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO scanned (name, location, rating)
                VALUES (%s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    rating = COALESCE(EXCLUDED.rating, scanned.rating),
                    location = EXCLUDED.location
            """, (game_name, location or None, rating))
        conn.commit()
    return {"ok": True}


@app.delete("/api/scanned/{game_name}")
def remove_scanned(game_name: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM scanned WHERE name = %s", (game_name,))
        conn.commit()
    return {"ok": True}


# --- Extra games ---

@app.get("/api/extra")
def get_extra():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name, year, location, rating FROM extra_games ORDER BY year, name")
            return [dict(r) for r in cur.fetchall()]


@app.post("/api/extra/scan")
async def scan_extra(
    file: UploadFile = File(...),
    location: Optional[str] = Form(None),
):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY nicht gesetzt")

    image_data = await file.read()
    texts = await call_vision_api(image_data)

    if not texts:
        return {"matched": False, "game": None}

    raw_name = texts[0].strip().split("\n")[0].strip()
    if not raw_name or len(raw_name) < 2:
        return {"matched": False, "game": None, "detected_text": " ".join(texts[:3])}

    import datetime
    year = datetime.datetime.now().year

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM extra_games WHERE LOWER(name) = LOWER(%s)", (raw_name,))
            existing = cur.fetchone()
            if not existing:
                cur.execute(
                    "INSERT INTO extra_games (name, year, location, rating) VALUES (%s, %s, %s, NULL)",
                    (raw_name, year, location or None)
                )
                new = True
            else:
                cur.execute("UPDATE extra_games SET location = %s WHERE LOWER(name) = LOWER(%s)", (location or None, raw_name))
                raw_name = existing["name"]
                new = False
        conn.commit()

    return {"matched": True, "game": raw_name, "new": new}


@app.post("/api/extra/rate/{game_name}")
def rate_extra(game_name: str, rating: Optional[int] = Form(None), location: Optional[str] = Form(None)):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE extra_games SET
                    rating = COALESCE(%s, rating),
                    location = %s
                WHERE name = %s
            """, (rating, location or None, game_name))
        conn.commit()
    return {"ok": True}


@app.post("/api/extra/rename/{game_name}")
def rename_extra(game_name: str, new_name: str = Form(...)):
    new_name = new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="Name darf nicht leer sein")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE extra_games SET name = %s WHERE name = %s", (new_name, game_name))
        conn.commit()
    return {"ok": True, "new_name": new_name}


@app.delete("/api/extra/{game_name}")
def delete_extra(game_name: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM extra_games WHERE name = %s", (game_name,))
        conn.commit()
    return {"ok": True}


@app.get("/admin", response_class=HTMLResponse)
def admin():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name, location, rating FROM scanned ORDER BY name")
            scanned = cur.fetchall()
            cur.execute("SELECT name, year, location, rating FROM extra_games ORDER BY year, name")
            extra = cur.fetchall()

    def stars(r):
        return ("★" * r + "☆" * (5 - r)) if r else "—"

    scanned_rows = "".join(
        f"<tr><td>{r['name']}</td><td>{r['location'] or '—'}</td><td>{stars(r['rating'])}</td></tr>"
        for r in scanned
    )
    extra_rows = "".join(
        f"<tr><td>{r['year']}</td><td>{r['name']}</td><td>{r['location'] or '—'}</td><td>{stars(r['rating'])}</td></tr>"
        for r in extra
    )

    return f"""
    <!DOCTYPE html>
    <html lang="de">
    <head>
      <meta charset="UTF-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
      <title>Admin — Eisenblätter's Spiele</title>
      <style>
        body {{ font-family: Segoe UI, sans-serif; background: #f5f0e8; padding: 2rem; }}
        h1 {{ color: #c0392b; margin-bottom: 0.3rem; }}
        h2 {{ color: #444; margin: 2rem 0 0.5rem; }}
        table {{ border-collapse: collapse; width: 100%; max-width: 700px; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        th {{ background: #c0392b; color: white; padding: 0.6rem 1rem; text-align: left; }}
        td {{ padding: 0.5rem 1rem; border-bottom: 1px solid #eee; }}
        tr:last-child td {{ border-bottom: none; }}
        .count {{ color: #888; font-size: 0.9rem; }}
      </style>
    </head>
    <body>
      <h1>🎲 Eisenblätter's Spiele — Admin</h1>

      <h2>🏆 Spiel des Jahres <span class="count">({len(scanned)} gesammelt)</span></h2>
      <table>
        <tr><th>Name</th><th>Standort</th><th>Bewertung</th></tr>
        {scanned_rows or '<tr><td colspan="3" style="color:#aaa">Noch keine Spiele</td></tr>'}
      </table>

      <h2>🎮 Weitere Spiele <span class="count">({len(extra)} gesamt)</span></h2>
      <table>
        <tr><th>Jahr</th><th>Name</th><th>Standort</th><th>Bewertung</th></tr>
        {extra_rows or '<tr><td colspan="4" style="color:#aaa">Noch keine Spiele</td></tr>'}
      </table>
    </body>
    </html>
    """


# Serve frontend in production
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
