from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from modules.ballot_creation.extract_symbol import ImageCrop
from modules.ballot_creation.resize_image import ImageResize
from fastapi.staticfiles import StaticFiles
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2


app = FastAPI()
crop_image = ImageCrop() 
resize_image = ImageResize()
app.mount("/static", StaticFiles(directory="./static"), name="static")


templates = Jinja2Templates(directory="./templates/CoolAdmin-master")


@app.get("/cropimage",response_class=HTMLResponse)
async def crop(request: Request):
    return templates.TemplateResponse("form.html",{"request":request})


@app.post("/cropimage",response_class=HTMLResponse) 
async def crop(files: list[UploadFile] = File(...)): 

    if files:
        for file in files: 
            if file.content_type in ["image/jpeg","image/png","image/jpg"]:
                print(file.filename)
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                crop_image.crop_symbol_from_image(img) 
            
            # image_suffix = Path(file.filename).suffix
            # with tempfile.NamedTemporaryFile(suffix='.jepg', delete=False)as image_temp:
            #     image_temp_path = image_temp.name
            #     print(image_temp_path)
            #     shutil.copyfileobj(image.file, image_temp)
            #     crop_image.crop_symbol_from_image(image_temp_path) 
            else:
                print(f"Invalid image format:{file.filename}")

    else:
        raise HTTPException(status_code=400, detail="Image file not recieved")


@app.get("/resizeimage",response_class=HTMLResponse)
async def crop(request: Request):
    return templates.TemplateResponse("resize_image.html",{"request":request})


@app.post("/resizeimage",response_class=HTMLResponse) 
async def crop(files: list[UploadFile] = File(...)): 

    images = [] 

    if files:
        for file in files:   
            print(file.content_type)                  
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)            
            images.append(img)       

        resize_image.resize_image(images)
    else: 
        raise HTTPException(status_code=400, detail="Image file not recieved")


@app.get("/",response_class=HTMLResponse)
async def crop(request: Request):
    return templates.TemplateResponse("predict.html",{"request":request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)