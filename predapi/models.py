from pydantic import BaseModel
from typing import List, Optional


class Signal2D(BaseModel):
    X: List
    y: Optional[int] = None
