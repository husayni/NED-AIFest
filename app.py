import os

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from broker import client, process_job

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello NED AI Fest!!"}


@app.post("/predict-async")
def predict(file: UploadFile = File(...)):
    import uuid

    filename = str(uuid.uuid4()) + ".png"
    filepath = os.path.join("data", filename)
    with open(filepath, "wb") as f:
        f.write(file.file.read())
    job_id = process_job.delay(filepath)
    return JSONResponse(
        status_code=201,
        content={"message": "Task Added To Queue", "job_id": str(job_id)},
    )


@app.get("/result/{job_id}")
def get_result(job_id: str):
    job = client.AsyncResult(job_id)
    if not job.ready():
        return JSONResponse(
            status_code=202, content={"job_id": str(job_id), "status": "Processing"}
        )
    result = job.get()
    return JSONResponse(
        status_code=200,
        content={"job_id": str(job_id), "status": "Success", "result": result},
    )
