import fastapi
import uvicorn
import tensorflow as tf

app = fastapi.FastAPI(
    title="Predict API",
    description="CNN for image classification",
    version="0.1",
    docs_url="/",
)


@app.post("/predict")
async def predict(X):
    model = app.state.model
    prediction = model.predict(X)
    return {"prediction": prediction}


@app.on_event("startup")
def load_model():
    model_file_path = "predapi/model.h5"
    model = tf.keras.models.load_model(model_file_path)
    app.state.model = model


if __name__ == "__main__":
    uvicorn.run(app)
