from kitbrc.utils.utils import parse_video_path_brc1, parse_video_path_brc2, get_file_hash
from kitbrc.data.video import Video, Camera
from kitbrc.annotations.bounding_boxes import Tracklets
import logging
from pathlib import Path
import pandas as pd
from uuid import uuid4
import json
from collections import defaultdict


class Subject:

    def __init__(self, subject_id: str):

        self.id = subject_id
        self.videos = []

    def add_video(self, video: Video):
        self.videos.append(video)


class Dataset:

    def __init__(self, video_manifest: str, camera_roi: str = None, camera_bbox_bounds: str = None, brc=1,
                 clip_id_to_filename=None,
                 subject_id_to_clip_id=None):

        self.cameras = {}
        self.brc = brc
        if camera_roi is not None:
            camera_roi = json.load(open(camera_roi))
            for camera, roi in camera_roi.items():
                if camera not in self.cameras:
                    self.cameras[camera] = Camera(camera)
                self.cameras[camera].roi = roi
        if camera_bbox_bounds is not None:
            camera_bbox_bounds = json.load(open(camera_bbox_bounds))
            for camera, bounds in camera_bbox_bounds.items():
                if camera not in self.cameras:
                    self.cameras[camera] = Camera(camera)
                self.cameras[camera].bbox_area_min = bounds[0]
                self.cameras[camera].bbox_area_max = bounds[1]
        if clip_id_to_filename is not None:
            file_id_mapping = {}
            for line in Path(clip_id_to_filename).read_text().strip().splitlines()[1:]:
                clip_id, filename = line.split(',')
                clip_id = int(clip_id)
                filename = Path(filename).name
                file_id_mapping[clip_id] = filename
        self.filename_to_subject_mapping = None
        if subject_id_to_clip_id is not None:
            self.filename_to_subject_mapping = defaultdict(list)
            for line in Path(subject_id_to_clip_id).read_text().strip().splitlines():
                line = line.split(',')
                subject_id = line[0]
                clip_ids = list(map(int, line[1:]))
                for clip_id in clip_ids:
                    self.filename_to_subject_mapping[file_id_mapping[clip_id]].append(
                        subject_id)
        if self.filename_to_subject_mapping is None and brc == 2:
            raise ValueError(
                'Must specify clip_id_to_filename and subject_id_to_clip_id files for BRC2')
        self.parse_dataset(video_manifest)

    def parse_dataset(self, video_manifest: str):

        self.subjects = {}
        self.video_map = {}
        videos = []
        for video_path in open(video_manifest).read().strip().splitlines():

            if self.brc == 1:
                try:
                    fields = parse_video_path_brc1(video_path)
                    if fields['subject_id'] not in self.subjects:
                        self.subjects[fields['subject_id']] = Subject(
                            fields['subject_id'])
                    subject = self.subjects[fields['subject_id']]
                    video_id = get_file_hash(Path(video_path).name)
                    fields['video_id'] = video_id
                    fields['camera'] = fields['camera']
                    if fields['camera'] not in self.cameras:
                        self.cameras[fields['camera']] = Camera(
                            camera_name=fields['camera'])
                    if 'field' in str(video_path):
                        fields['tracklets'] = Tracklets(
                            Path(video_path.replace('.mp4', '.csv')), self.cameras[fields['camera']], brc=self.brc)
                    elif 'controlled' in str(video_path):
                        fields['tracklets'] = Tracklets(
                            Path(video_path.replace('.mp4', '.mp4.wb.csv')), self.cameras[fields['camera']], brc=self.brc)
                    else:
                        raise NotImplementedError

                    video = Video(**fields)
                    subject.add_video(video)
                    videos.append(video)
                    self.video_map[video_id] = video

                except ValueError as e:

                    logging.warn(
                        'Failed to parse video file: '+str(video_path))
            elif self.brc == 2:
                try:
                    fields_list = parse_video_path_brc2(
                        video_path, self.filename_to_subject_mapping)
                    for fields in fields_list:
                        if fields['subject_id'] not in self.subjects:
                            self.subjects[fields['subject_id']] = Subject(
                                fields['subject_id'])
                        subject = self.subjects[fields['subject_id']]
                        video_id = uuid4()
                        fields['video_id'] = video_id
                        fields['camera'] = fields['camera']
                        if fields['camera'] not in self.cameras:
                            self.cameras[fields['camera']] = Camera(
                                camera_name=fields['camera'])
                        if 'field' in str(video_path):
                            fields['tracklets'] = Tracklets(
                                Path(video_path.replace('.mp4', '.csv')), self.cameras[fields['camera']], brc=self.brc)
                        elif 'brc2-enrollment' in str(video_path):
                            if 'wholebody' in str(video_path):
                                continue
                            elif 'gait' in str(video_path):
                                fields['tracklets'] = Tracklets(
                                    Path(video_path.replace('.mp4', '.csv')), self.cameras[fields['camera']], brc=self.brc)
                            else:
                                raise NotImplementedError
                        else:
                            raise NotImplementedError

                        video = Video(**fields)
                        subject.add_video(video)
                        videos.append(video)
                        self.video_map[video_id] = video

                except ValueError as e:
                    # print(e)
                    logging.warn(
                        'Failed to parse video file: '+str(video_path))
            else:
                raise NotImplementedError

        self.videos = pd.DataFrame(videos)
