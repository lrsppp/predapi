import fastapi
import uvicorn

from predapi.models import Image

app = fastapi.FastAPI(
    title="Predict API",
    description="CNN for image classification",
    version="0.1",
    docs_url="/",
)


@app.get("/predict")
def predict(image: Image):
    return {"class": Image.text}


uvicorn.run(app)
