import os
import json
import base64
import httpx
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GAMES_FILE = Path(__file__).parent / "games.json"
SCANNED_FILE = Path(__file__).parent / "scanned.json"
EXTRA_FILE = Path(__file__).parent / "extra_games.json"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

with open(GAMES_FILE) as f:
    GAMES = json.load(f)


def load_scanned() -> dict:
    if SCANNED_FILE.exists():
        with open(SCANNED_FILE) as f:
            data = json.load(f)
            # migrate old list format to dict
            if isinstance(data, list):
                return {name: {"location": None, "rating": None} for name in data}
            return data
    return {}


def save_scanned(scanned: dict):
    with open(SCANNED_FILE, "w") as f:
        json.dump(scanned, f, ensure_ascii=False, indent=2)


def load_extra() -> list:
    if EXTRA_FILE.exists():
        with open(EXTRA_FILE) as f:
            return json.load(f)
    return []


def save_extra(extra: list):
    with open(EXTRA_FILE, "w") as f:
        json.dump(extra, f, ensure_ascii=False, indent=2)


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
    scanned = load_scanned()
    return [
        {
            **game,
            "scanned": game["name"] in scanned,
            "location": scanned.get(game["name"], {}).get("location"),
            "rating": scanned.get(game["name"], {}).get("rating"),
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

    scanned = load_scanned()
    scanned[matched] = {
        "location": location,
        "rating": scanned.get(matched, {}).get("rating"),
    }
    save_scanned(scanned)

    return {"matched": True, "game": matched}


@app.post("/api/rate/{game_name}")
def rate_game(game_name: str, rating: int = Form(...)):
    if rating < 1 or rating > 5:
        raise HTTPException(status_code=400, detail="Bewertung muss zwischen 1 und 5 sein")
    scanned = load_scanned()
    if game_name not in scanned:
        scanned[game_name] = {"location": None, "rating": rating}
    else:
        scanned[game_name]["rating"] = rating
    save_scanned(scanned)
    return {"ok": True}


@app.delete("/api/scanned/{game_name}")
def remove_scanned(game_name: str):
    scanned = load_scanned()
    if game_name in scanned:
        del scanned[game_name]
        save_scanned(scanned)
    return {"ok": True}


# --- Extra games (non-Spiel des Jahres) ---

@app.get("/api/extra")
def get_extra():
    return load_extra()


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

    # Use the first detected text block as the game name
    raw_name = texts[0].strip().split("\n")[0].strip()
    if not raw_name or len(raw_name) < 2:
        return {"matched": False, "game": None, "detected_text": " ".join(texts[:3])}

    extra = load_extra()
    existing = next((g for g in extra if g["name"].lower() == raw_name.lower()), None)
    if not existing:
        import datetime
        new_game = {
            "name": raw_name,
            "year": datetime.datetime.now().year,
            "location": location,
            "rating": None,
        }
        extra.append(new_game)
        save_extra(extra)
        return {"matched": True, "game": raw_name, "new": True}
    else:
        existing["location"] = location
        save_extra(extra)
        return {"matched": True, "game": raw_name, "new": False}


@app.post("/api/extra/rate/{game_name}")
def rate_extra(game_name: str, rating: int = Form(...)):
    if rating < 1 or rating > 5:
        raise HTTPException(status_code=400, detail="Bewertung muss zwischen 1 und 5 sein")
    extra = load_extra()
    game = next((g for g in extra if g["name"] == game_name), None)
    if not game:
        raise HTTPException(status_code=404, detail="Spiel nicht gefunden")
    game["rating"] = rating
    save_extra(extra)
    return {"ok": True}


@app.delete("/api/extra/{game_name}")
def delete_extra(game_name: str):
    extra = load_extra()
    extra = [g for g in extra if g["name"] != game_name]
    save_extra(extra)
    return {"ok": True}


# Serve frontend in production
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
