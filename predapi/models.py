from pydantic import BaseModel, BaseSettings
from typing import List, Optional


class ModelConfig(BaseSettings):
    num_filters: int
    filter_size: int
    pool_size: int


class TrainConfig(BaseSettings):
    epochs: int


class Signal2D(BaseModel):
    X: List
    y: Optional[int] = None
