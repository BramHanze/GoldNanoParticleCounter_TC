from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List
from pathlib import Path
import tempfile
import os
import json
import yaml

from blackdotdetector import BlackDotDetector
#from ..filemanager import FileManager

app = FastAPI()

# Output directory path
output_folder = Path("Frontend/output/")

# Serve output directory statically
app.mount("/output/", StaticFiles(directory=str(output_folder)), name="output")

@app.post("/detect_dots/")
async def get_dots(
    min_area: float = Form(...),
    circ_threshold: float = Form(...),
    cell_min_area: float = Form(...),
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
                image_path=str(path),
                min_area=min_area,
                circularity_threshold=circ_threshold
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

@app.get("/get_yaml")
async def get_yaml():
    yaml_path = Path(__file__).parent / "config.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return JSONResponse(content=data)
    else:
        raise HTTPException(status_code=404, detail="YAML file not found.")

@app.post("/update_yaml")
async def update_yaml(request: Request):
    new_data = await request.json()
    yaml_path = Path(__file__).parent / "config.yaml"

    try:
        with open(yaml_path, "w") as f:
            yaml.safe_dump(new_data, f)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to update YAML: {str(e)}")

@app.get("/{page_name}")
async def serve_html_page(page_name: str):
    html_path = Path(__file__).parent / f"{page_name}.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="Page not found.")
