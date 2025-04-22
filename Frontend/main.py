from fastapi import FastAPI, UploadFile, File, Form, Response
from pydantic import BaseModel
from blackdotdetector import BlackDotDetector

"""
Data class for the input parameters
"""
class Item(BaseModel):
    min_area: float
    circ_threshold: float
    cell_min_area: float
    image_file: UploadFile = File(...)

app = FastAPI()

@app.post("/detect_dots/")
async def get_dots(
    data: Item = Form(...)):

    image_contents = await data.image_file.read()

    detector = BlackDotDetector(
        image_path=data.image_file.filename,
        image_data=image_contents,
        min_area=data.min_area,
        circularity_threshold=data.circ_threshold,
        cell_min_area=data.cell_min_area
    )

    img_buf = detector.run()
    bufContents: bytes = img_buf.getvalue()
    
    headers = {'Content-Disposition': 'inline; filename="mask.png"'}
    return Response(bufContents, headers=headers, media_type='image/png')
