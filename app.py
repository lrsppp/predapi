import fastapi
import uvicorn

from predapi.models import Image
from predapi.cnn import SimpleCNN

app = fastapi.FastAPI(
    title="Predict API",
    description="CNN for image classification",
    version="0.1",
    docs_url="/",
)
cnn_obj = SimpleCNN(8, 2, 2)


@app.get("/predict")
def predict(image: Image):
    return {"class": Image.text}


@app.on_event("startup")
def load_model():
    model_path = "model.h5"
    cnn_obj.load_model(model_path)


if __name__ == "__main__":
    uvicorn.run(app)
