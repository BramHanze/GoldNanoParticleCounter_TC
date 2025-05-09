from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List
from pathlib import Path
import tempfile
import zipfile
import io
import os
import json

from ..Backend.blackdotdetector import BlackDotDetector

app = FastAPI()

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
         output_path = f'output/{path.stem}.json'
         if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
               data = json.load(f)
               results.append({"image": path.stem, **data})
      except Exception as e:
         print(f"Error processing {path.name}: {str(e)}")

   return JSONResponse(content={"results": results})

@app.get("/")
async def serve_client_page():
   client_html_path = Path(__file__).parent / "client.html"
   if client_html_path.exists():
      return HTMLResponse(content=client_html_path.read_text(), media_type="text/html")
