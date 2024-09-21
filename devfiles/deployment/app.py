from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from typing import List

app = FastAPI()

# Mount static files (for stylesheets, JS, etc., if needed)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up template rendering
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def get_upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...), description: str = Form(...)):
    file_location = f"uploaded_{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    return templates.TemplateResponse("upload.html", {
        "request": request,
        "filename": file.filename,
        "description": description,
        "file_location": file_location
    })


@app.post("/upload-multiple/")
async def upload_multiple_files(request: Request, files: List[UploadFile] = File(...)):
    uploaded_files = []
    for file in files:
        file_location = f"uploaded_{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
        uploaded_files.append(file.filename)

    return templates.TemplateResponse("upload.html", {
        "request": request,
        "uploaded_files": uploaded_files
    })
