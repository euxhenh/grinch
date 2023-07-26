from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel


class Report(BaseModel):

    model_config = {
        'arbitrary_types_allowed': True,
        'extra': 'forbid',
    }

    cls: str
    config: Dict[str, Any] | None = None
    message: Optional[str] = None
    shape: Optional[Tuple[int, int]] = None
    artifacts: Optional[str | List[str]] = None


class Reporter:
    """A class that logs information from the difference steps of the
    pipeline. Can be useful for reproducibility purposes.
    """
    def __init__(self):
        self.counter: int = 0

    def log(self, report: Report):
        rep = report.model_dump(exclude_none=True)  # noqa
