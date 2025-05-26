from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse,Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2
import os
import json
from torchvision.transforms.functional import to_tensor 


# from modules.ballot_creation.extract_symbol import ImageCrop
# from modules.ballot_creation.resize_image import ImageResize
from modules.symbol_detection.faster_rcnn.pipeline.prediction import PredictionPipeline


app = FastAPI()
# crop_image = ImageCrop() 
# resize_image = ImageResize()
predict_obj = PredictionPipeline()


app.mount("/static", StaticFiles(directory="./static"), name="static")
app.mount("/processed_image", StaticFiles(directory="./static/vote_results/visualize"), name="processed_image")



templates = Jinja2Templates(directory="./templates/CoolAdmin-master")

def load_json(filename="vote.json"):
    with open(filename, "r") as f:
        return json.load(f)


@app.get("/",response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("base.html",{"request":request})


@app.get("/display_prediction",response_class=HTMLResponse)
async def display_prediction(request: Request,processed_image: str = None):
    try:
        json_data = load_json()
    except FileNotFoundError:
        json_data = {}
    return templates.TemplateResponse("display-prediction.html",{"request":request,"processed_image": processed_image, "data": json_data})


@app.get("/select_ballot",response_class=HTMLResponse)
async def select_ballot(request: Request,processed_image: str = None):
    return templates.TemplateResponse("select-ballot.html",{"request":request,"processed_image": processed_image})
 


@app.post("/process_ballot",response_class=HTMLResponse) 
async def process_ballot(image_file: UploadFile = File(...)): 

    images = [] 
    if not image_file or not image_file.filename:
        return RedirectResponse(url="/?error=no_file", status_code=303)
    try:   
        if image_file.content_type not in ['image/jpeg', 'image/png']:
            return RedirectResponse(url="/?error=invalid_type", status_code=303)


        # print(image_file.content_type) 
        # print(f"Processing file: {image_file.filename}, Cont6ent-Type: {image_file.content_type}")
        # save_path = os.path.join("./static/images/predict_image", image_file)

        # print(f"Saving file to: {save_path}")  # Debugging print
        image_name = image_file.filename
        contents = await image_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   
        if img is None:
                raise ValueError("Invalid image format")     

        image_tensor = to_tensor(img)    
        image_tensor = image_tensor.unsqueeze(0)   
        
        predict_obj.predict(image_tensor)
        predict_obj.visualize(img, image_name)
        predict_obj.validate_vote(img,image_name)

        return RedirectResponse(url=f"/display_prediction?processed_image=processed_image/{image_file.filename}", status_code=303)


    except Exception as e: 
        return RedirectResponse(url="/?error=processing_error", status_code=303)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)