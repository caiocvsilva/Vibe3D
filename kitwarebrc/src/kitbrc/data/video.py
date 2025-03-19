from __future__ import annotations
from dataclasses import dataclass, field
from kitbrc.annotations.bounding_boxes import Tracklets
from pathlib import Path
from datetime import datetime
from uuid import UUID
from typing import Tuple


@dataclass()
class Camera:
    camera_name: str
    bbox_area_min: int = None
    bbox_area_max: int = None
    roi: Tuple[int] = None


@dataclass()
class Video:
    video_id: UUID
    subject_id: str
    camera: str
    type: str
    location: str
    set_num: int
    activity: str
    tracklets: Tracklets
    file_path: Path
    time_start: datetime
    time_stop: datetime
