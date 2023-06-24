import fastapi
import uvicorn
from tensorflow.keras.models import load_model
from predapi.models import Signal2D

app = fastapi.FastAPI(
    title="Predict API",
    description="API for CNN for Image Classification",
    version="0.1",
    docs_url="/",
)


@app.post("/predict")
def predict(signal2d: Signal2D):
    model = app.state.model
    prediction = model.predict(signal2d.X).argmax()
    return {"prediction": int(prediction)}


@app.on_event("startup")
def load_cnn():
    model_file_path = "model/model.h5"
    model = load_model(model_file_path)
    app.state.model = model


if __name__ == "__main__":
    uvicorn.run(app)
