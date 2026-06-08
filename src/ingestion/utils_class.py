from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

@dataclass
class ImageInfo:
    node_id: str
    path: Path
    page: int
    bbox: Tuple[float, float, float, float]


@dataclass
class ProcessingResult:
    markdown: str
    images: List[ImageInfo] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

@dataclass
class CrossReference:
    target_node_id: str
    reference_type: str
    context: str
    page: Optional[int]