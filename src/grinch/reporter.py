from typing import List, Optional, Tuple

from pydantic import BaseModel, Extra


class Report(BaseModel):
    cls: str
    config: Optional[BaseModel] = None
    message: Optional[str] = None
    shape: Optional[Tuple[int, int]] = None
    artifacts: Optional[str | List[str]] = None

    class Config:
        arbitrary_types_allowed = True
        smart_union = True
        extra = Extra.forbid


class Reporter:
    """A class that logs information from the difference steps of the
    pipeline. Can be useful for reproducibility purposes.
    """
    def __init__(self):
        self.counter: int = 0

    def log(self, report: Report):
        rep = report.dict(exclude_none=True)  # noqa
