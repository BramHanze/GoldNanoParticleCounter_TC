from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.responses import HTMLResponse
from typing import List
from pathlib import Path
import tempfile
import zipfile
import io
import shutil
import os

from blackdotdetector import BlackDotDetector

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

   for upload in image_file:
      file_path = temp_dir / upload.filename
      file_path.parent.mkdir(parents=True, exist_ok=True)
      with open(file_path, "wb") as f:
         content = await upload.read()
         f.write(content)
         saved_files.append(file_path)

   processed_images = []
   for path in saved_files:
      try:
         detector = BlackDotDetector(
            image_path=str(path),
            min_area=min_area,
            circularity_threshold=circ_threshold
         )

         img_buf = detector.run()

         if img_buf:
            processed_images.append((f"{path.stem}_processed.png", img_buf))
         else:
            print(f"Warning: run() returned None for {path.name}")

      except Exception as e:
         print(f"Error processing {path.name}: {e}")

   if processed_images:
      zip_buf = io.BytesIO()
      with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zipf:
         for name, buf in processed_images:
            try:
               zipf.writestr(name, buf.getvalue())
            except AttributeError:
               print(f"Invalid buffer for {name}, skipping.")
      zip_buf.seek(0)
      headers = {
         "Content-Disposition": "attachment; filename=processed_images.zip"
      }
      return Response(content=zip_buf.getvalue(), media_type="application/zip", headers=headers)

   return Response("No valid images processed.", media_type="text/plain", status_code=400)

@app.get("/")
async def serve_client_page():
   client_html_path = Path(__file__).parent / "client.html"
   if client_html_path.exists():
      return HTMLResponse(content=client_html_path.read_text(), media_type="text/html")
