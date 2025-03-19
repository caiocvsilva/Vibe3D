from pathlib import Path
import pandas as pd
from kitbrc.utils import intersection_over_union, parse_dive_annotations
import numpy as np
from filterpy.kalman import KalmanFilter
from collections import defaultdict


class Tracklets:

    def __init__(self, tracklet_file: Path, camera: 'Camera', brc=1):

        self.tracklet_file = tracklet_file
        self.camera = camera
        self.brc = brc

    @property
    def exists(self):
        return self.tracklet_file.is_file()

    @property
    def data(self):

        if self.exists:
            tracklet_file_text = self.tracklet_file.read_text()
            return parse_dive_annotations(tracklet_file_text, brc=self.brc)
        return None

    def filter(self, class_id=None):
        annotations = self.data
        if class_id is not None:
            annotations = annotations[annotations['class'] == class_id]
        TL_x = annotations.TL_x.values
        TL_y = annotations.TL_y.values
        BR_x = annotations.BR_x.values
        BR_y = annotations.BR_y.values

        # Filter Annotations based on distribution of human sized
        # bboxes per camera
        annotations_area = np.abs(BR_x-TL_x)*np.abs(BR_y-TL_y)
        annotations_mask = np.ones(len(annotations), dtype=bool)
        if self.camera.bbox_area_min is not None:
            annotations_mask &= annotations_area > self.camera.bbox_area_min
        if self.camera.bbox_area_max is not None:
            annotations_mask &= annotations_area < self.camera.bbox_area_max
        cx = (TL_x + BR_x) / 2
        cy = (TL_y + BR_y) / 2

        # Filter Annotations to only those in ROI
        if self.camera.roi is not None:
            x1, y1, x2, y2 = self.camera.roi
            xmax, xmin = max(x1, x2), min(x1, x2)
            ymax, ymin = max(y1, y2), min(y1, y2)
            annotations_mask &= (cx > xmin) & (
                cx < xmax) & (cy > ymin) & (cy < ymax)
        annotations = annotations[annotations_mask]

        # Merge Tracks with IoU above threshold
        # Keep only largest track
        iou_thresh = 0.7
        for tracklet in list(annotations.track_id.unique()):
            tracklet_df = annotations[annotations.track_id == tracklet]
            sub_df = pd.merge(tracklet_df, annotations, on='frame')
            for contemporary_tracklet in sub_df.track_id_y.unique():
                if contemporary_tracklet == tracklet:
                    continue
                sub_df_con = sub_df[sub_df.track_id_y == contemporary_tracklet]
                # Compute IOU
                p1_x = sub_df_con.TL_x_x.values
                p1_y = sub_df_con.TL_y_x.values
                p2_x = sub_df_con.BR_x_x.values
                p2_y = sub_df_con.BR_y_x.values

                p3_x = sub_df_con.TL_x_y.values
                p3_y = sub_df_con.TL_y_y.values
                p4_x = sub_df_con.BR_x_y.values
                p4_y = sub_df_con.BR_y_y.values

                p1 = [(x, y) for x, y in zip(p1_x, p1_y)]
                p2 = [(x, y) for x, y in zip(p2_x, p2_y)]
                p3 = [(x, y) for x, y in zip(p3_x, p3_y)]
                p4 = [(x, y) for x, y in zip(p4_x, p4_y)]

                iou = list(map(intersection_over_union, zip(p1, p2, p3, p4)))
                if np.mean(iou) > iou_thresh:
                    annotations.track_id[annotations.track_id ==
                                         contemporary_tracklet] = tracklet

        track_priority = annotations.track_id.value_counts()

        TL_x = annotations.TL_x.values
        TL_y = annotations.TL_y.values
        BR_x = annotations.BR_x.values
        BR_y = annotations.BR_y.values

        cx = (TL_x + BR_x) / 2
        cy = (TL_y + BR_y) / 2
        annotations['TL_x'] = TL_x
        annotations['TL_y'] = TL_y
        annotations['BR_x'] = BR_x
        annotations['BR_y'] = BR_y
        annotations['w'] = np.abs(BR_x-TL_x)
        annotations['h'] = np.abs(TL_y-BR_y)
        annotations['centroid_x'] = cx
        annotations['centroid_y'] = cy

        centroid_filter_x = None
        centroid_filter_y = None
        height_filter = None
        width_filter = None
        filtered_annotations = defaultdict(list)
        inferred = []
        inferred_count = 0
        try:
            for cnt in range(int(annotations.index.min()), int(annotations.index.max())+1):
                if cnt in annotations.index:
                    # Get all detections for frame
                    rows = annotations.loc[cnt]
                    if isinstance(rows, pd.DataFrame):
                        # We have multiple detections for this frame
                        # Choose most confident detection from the longest track
                        track_ids = rows.track_id.unique()
                        priority_level = track_priority.loc[track_ids]
                        longest_track = priority_level.index[priority_level.argmax(
                        )]
                        rows = rows[rows.track_id == longest_track]
                        most_confident = np.argmax(rows.confidence.values)
                        rows = rows.iloc[most_confident]

                    # Grab detection measurement
                    w = rows.w
                    h = rows.h
                    cx = rows.centroid_x
                    cy = rows.centroid_y

                    if centroid_filter_x is None:
                        # We need to initialize the Kalman Filter

                        centroid_filter_x = KalmanFilter(dim_x=2, dim_z=1)
                        centroid_filter_y = KalmanFilter(dim_x=2, dim_z=1)
                        height_filter = KalmanFilter(dim_x=2, dim_z=1)
                        width_filter = KalmanFilter(dim_x=2, dim_z=1)

                        centroid_filter_x.x = np.array([cx, 0.])
                        centroid_filter_x.F = np.array([[1., 1.],
                                                        [0., 1.]])
                        centroid_filter_x.H = np.array([[1., 0.]])
                        centroid_filter_x.P *= 500.
                        centroid_filter_x.R = 5000

                        centroid_filter_y.x = np.array([cy, 0.])
                        centroid_filter_y.F = np.array([[1., 1.],
                                                        [0., 1.]])
                        centroid_filter_y.H = np.array([[1., 0.]])
                        centroid_filter_y.P *= 500.
                        centroid_filter_y.R = 5000

                        height_filter.x = np.array([h, 0.])
                        height_filter.F = np.array([[1., 1.],
                                                    [0., 1.]])
                        height_filter.H = np.array([[1., 0.]])
                        height_filter.P *= 500.
                        height_filter.R = 5000

                        width_filter.x = np.array([w, 0.])
                        width_filter.F = np.array([[1., 1.],
                                                   [0., 1.]])
                        width_filter.H = np.array([[1., 0.]])
                        width_filter.P *= 500.
                        width_filter.R = 5000
                        x1 = rows.TL_x
                        y1 = rows.TL_y
                        x2 = rows.BR_x
                        y2 = rows.BR_y

                    else:
                        # Update the filter with new measurement
                        centroid_filter_x.predict()
                        centroid_filter_y.predict()
                        height_filter.predict()
                        width_filter.predict()

                        centroid_filter_x.update([cx])
                        centroid_filter_y.update([cy])
                        height_filter.update([h])
                        width_filter.update(w)

                        cx = centroid_filter_x.x[0]
                        cy = centroid_filter_y.x[0]
                        h = height_filter.x[0]
                        w = width_filter.x[0]

                        x1 = int(cx-w/2)
                        x2 = int(cx+w/2)
                        y1 = int(cy-h/2)
                        y2 = int(cy+h/2)

                    for column, value in dict(rows).items():
                        filtered_annotations[column].append(value)
                    filtered_annotations['TL_x'][-1] = x1
                    filtered_annotations['BR_x'][-1] = x2
                    filtered_annotations['TL_y'][-1] = y1
                    filtered_annotations['BR_y'][-1] = y2
                    filtered_annotations['frame'].append(cnt)
                    inferred.append(False)
                    inferred_count = 0
                else:
                    # We do not have a measurement for this frame
                    if centroid_filter_x is None:
                        # This only happends if we have
                        # no detections before this frame
                        # so nothing to filter
                        continue
                    if inferred_count > 5:
                        centroid_filter_x = None
                        centroid_filter_y = None
                        height_filter = None
                        width_filter = None
                        continue

                    centroid_filter_x.predict()
                    centroid_filter_y.predict()
                    height_filter.predict()
                    width_filter.predict()

                    cx = centroid_filter_x.x
                    cy = centroid_filter_y.x
                    h = height_filter.x
                    w = width_filter.x

                    cx = centroid_filter_x.x[0]
                    cy = centroid_filter_y.x[0]
                    h = height_filter.x[0]
                    w = width_filter.x[0]

                    x1 = int(cx-w/2)
                    x2 = int(cx+w/2)
                    y1 = int(cy-h/2)
                    y2 = int(cy+h/2)

                    for col in filtered_annotations.keys():
                        filtered_annotations[col].append(None)
                    filtered_annotations['TL_x'][-1] = x1
                    filtered_annotations['BR_x'][-1] = x2
                    filtered_annotations['TL_y'][-1] = y1
                    filtered_annotations['BR_y'][-1] = y2
                    filtered_annotations['frame'][-1] = cnt
                    inferred.append(True)
                    inferred_count += 1

            # Remove intermediate columns
            for col in ['w', 'h', 'centroid_x', 'centroid_y']:
                del filtered_annotations[col]
            filtered_annotations['inferred'] = inferred
            return pd.DataFrame(filtered_annotations).set_index('frame')
        except Exception as e:
            return None
