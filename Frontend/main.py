from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List
from pathlib import Path
import tempfile
import yaml
import json
import cv2
import pickle

from ..Backend.scale_finder import scale_finder
from ..Backend.blackdotdetector import BlackDotDetector
#from ..Backend.filemanager import FileManager

app = FastAPI()

config = yaml.safe_load(open("Frontend/config.yml"))
# Output directory path
output_folder = Path(config['output_directory'])

# Tags file path
tags_file = Path(config['output_directory']+"/Tags/tags.json")

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
        if config['only_run_new_images']:
            # Check if any output JSON exists for this image name (regardless of extension)
            output_json = output_folder / f"{path.stem}.json"
            if output_json.exists():
                print(f"Skipping {path.name}: already processed (found {output_json.name}).")
                with open(output_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append({"image": path.stem, **data})
                continue
        
        if config['predict_best_settings']:
            image = cv2.imread(path)
            height = image.shape[0:2][0]
            scale = scale_finder(path)

            model = pickle.load(open("Backend\model.pkl", "rb"))
            min_area, dot_blur = model.predict([[scale, height]])[0]
            min_area = float(min_area)
            dot_blur = int(dot_blur)
            if  dot_blur % 2 == 0: # make dot_blur is odd if not already
                dot_blur += 1
            
            try:
                detector = BlackDotDetector(
                    image_path=str(path), min_area=min_area, dot_blur=dot_blur, scale=scale,
                )
                detector.run()
            except Exception as e:
                print(f"Error processing {path.name}: {str(e)}")
        else:
            try:
                detector = BlackDotDetector(
                    image_path=str(path),
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

@app.get("/get_yaml")
async def get_yaml():
    yaml_path = Path(__file__).parent / "config.yml"
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return JSONResponse(content=data)
    else:
        raise HTTPException(status_code=404, detail="YAML file not found.")

@app.post("/update_yaml")
async def update_yaml(request: Request):
    new_data = await request.json()
    yaml_path = Path(__file__).parent / "config.yml"
    try:
        with open(yaml_path, "w") as f:
            yaml.dump(new_data, f, default_flow_style=None)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to update YAML: {str(e)}")

@app.get("/get_default_yaml")
async def get_default_yaml():
    yaml_path = Path(__file__).parent / "config_default.yml"
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return JSONResponse(content=data)
    else:
        raise HTTPException(status_code=404, detail="Default YAML file not found.")

@app.get("/get_image/{image_name}")
def get_image(image_name: str):
    for ext in ("jpg", "jpeg", "png"):
        candidate = output_folder / f"{image_name}.{ext}"
        if candidate.is_file():
            media_type = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
            return FileResponse(str(candidate), media_type=media_type)
    raise HTTPException(status_code=404, detail="Image not found.")

@app.get("/list_previous_images/")
def list_previous_images():
    images = [f.stem for f in output_folder.glob("*.jpg")]
    images += [f.stem for f in output_folder.glob("*.png")]
    images += [f.stem for f in output_folder.glob("*.jpeg")]
    return JSONResponse(content={"images": images})

@app.post("/delete_results/")
async def delete_results(request: Request):
    data = await request.json()
    images = data.get("images", [])
    deleted_files = []

    for image in images:
        for ext in [".jpg", ".png", ".jpeg", ".json"]:
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

@app.get("/get_tags/")
async def get_tags():
    if tags_file.exists():
        with open(tags_file, 'r', encoding='utf-8') as f:
            tags = json.load(f)
        return JSONResponse(content={"tags": tags})
    return JSONResponse(content={"tags": []})

@app.post("/assign_tags/")
async def assign_tags(request: Request):
    data = await request.json()
    image = data.get("image")
    tags = data.get("tags", [])
    json_path = output_folder / f"{image}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Image JSON not found.")
    with open(json_path, 'r', encoding='utf-8') as f:
        image_data = json.load(f)
    image_data['tags'] = tags
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(image_data, f, indent=2)
    return JSONResponse(content={"message": "Tags assigned."})

@app.post("/adjust_dots/")
async def adjust_dots(request: Request):
    data = await request.json()
    image = data.get("image")
    adjust = data.get("adjust", 0)
    json_path = output_folder / f"{image}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Image JSON not found.")
    with open(json_path, 'r', encoding='utf-8') as f:
        image_data = json.load(f)
    image_data['adjust_dots'] = adjust
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(image_data, f, indent=2)
    return JSONResponse(content={"message": "Adjustment saved."})

@app.get("/{page_name}")
async def serve_html_page(page_name: str):
    html_path = Path(__file__).parent / f"{page_name}.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), media_type="text/html")