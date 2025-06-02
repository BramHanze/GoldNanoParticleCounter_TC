from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List
from pathlib import Path
import tempfile
import os
import json

from ..Backend.blackdotdetector import BlackDotDetector
from ..Backend.filemanager import FileManager

app = FastAPI()

# Output directory path
output_folder = Path(r"D:\Rene\Documents\school\goudbolletjes\output")

# Tags file path
tags_file = Path(r"D:\Rene\Documents\school\goudbolletjes\Tags\tags.json")

# Serve output directory statically
app.mount("/output", StaticFiles(directory=str(output_folder)), name="output")

@app.post("/detect_dots/")
async def get_dots(
    image_file: List[UploadFile] = File(...)
):
    temp_dir = Path(tempfile.mkdtemp(prefix="uploaded_folder_"))
    print(f"Uploaded files will be saved to: {temp_dir}")

    saved_files = []
    results = []

    for upload in image_file:
        file_path = temp_dir / upload.filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            content = await upload.read()
            f.write(content)
            saved_files.append(file_path)

    for path in saved_files:
        try:
            detector = BlackDotDetector(
                image_path=str(path)
            )
            detector.run()
            output_path = output_folder / f"{path.stem}.json"
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append({"image": path.stem, **data})
        except Exception as e:
            print(f"Error processing {path.name}: {str(e)}")

    return JSONResponse(content={"results": results})


@app.get("/get_image/{image_name}")
def get_image(image_name: str):
    image_path = output_folder / f"{image_name}.jpg"
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(str(image_path), media_type="image/jpeg")


@app.get("/list_previous_images/")
def list_previous_images():
    images = [f.stem for f in output_folder.glob("*.jpg")]
    return JSONResponse(content={"images": images})


@app.post("/delete_results/")
async def delete_results(request: Request):
    data = await request.json()
    images = data.get("images", [])
    deleted_files = []

    for image in images:
        for ext in [".jpg", ".json"]:
            file_path = output_folder / f"{image}{ext}"
            if file_path.exists():
                try:
                    file_path.unlink()
                    deleted_files.append(str(file_path.name))
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

    return JSONResponse(content={"deleted": deleted_files})


@app.get("/")
async def serve_client_page():
    client_html_path = Path(__file__).parent / "client.html"
    if client_html_path.exists():
        return HTMLResponse(content=client_html_path.read_text(), media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="Client page not found.")
    
@app.post("/add_tag/")
async def add_tag(request: Request):
    data = await request.json()
    tag = data.get("tag", "").strip()

    if not tag:
        raise HTTPException(status_code=400, detail="No tag provided")

    # Load existing tags
    tags = []
    if tags_file.exists():
        try:
            with open(tags_file, 'r', encoding='utf-8') as f:
                tags = json.load(f)
        except json.JSONDecodeError:
            tags = []

    # Avoid duplicates
    if tag in tags:
        return JSONResponse(content={"message": "Tag already exists"}, status_code=200)

    # Add new tag and save
    tags.append(tag)
    with open(tags_file, 'w', encoding='utf-8') as f:
        json.dump(tags, f, indent=2)

    return JSONResponse(content={"message": "Tag added successfully"}, status_code=200)
