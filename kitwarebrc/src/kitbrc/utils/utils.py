from pathlib import Path
from typing import Union
from datetime import datetime
import numpy as np
import pandas as pd
from io import StringIO
from hashlib import sha256


def get_file_hash(file_path):

    file_name = Path(file_path).name
    return sha256(file_name.encode('utf-8')).hexdigest()[:8]


def parse_video_path_brc1(file_path: Union[str, Path]) -> dict:

    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not isinstance(file_path, Path):
        raise TypeError
    if not file_path.exists() or file_path.suffix != '.mp4':
        raise ValueError

    # At this point we have a video file that exists

    file_name = file_path.stem.split('_')
    if 'field' in str(file_path):
        fields = {
            'type': 'field',
            'activity': file_name[2],
            'camera': file_name[3],
            'location': file_name[4],
            'time_start': datetime.strptime('_'.join(file_name[5:7]), '%Y-%m-%d_%H-%M-%S')
        }
    elif 'controlled' in str(file_path):
        if len(file_path.suffixes) < 2:
            fields = {
                'type': 'controlled',
                'activity': 'untrimmed',
                'camera': file_name[3],
                'location': 'gait_enrollment',
                'time_start': None
            }
        if len(file_path.suffixes) == 2:
            fields = {
                'type': 'controlled',
                'activity': '_'.join(file_name[2:3]),
                'camera': file_path.suffixes[0][1:],
                'location': 'gait_enrollment',
                'time_start': None
            }

    else:
        raise ValueError

    fields['subject_id'] = file_name[0]
    fields['set_num'] = int(file_name[1][3:])
    fields['file_path'] = file_path
    fields['time_stop'] = None

    return fields


def parse_video_path_brc2(file_path: Union[str, Path], filename_to_subject_id_mapping) -> dict:

    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not isinstance(file_path, Path):
        raise TypeError
    if not file_path.exists() or file_path.suffix != '.mp4':
        raise ValueError

    # At this point we have a video file that exists

    file_name = file_path.stem.split('_')
    if 'field' in str(file_path):
        tracklet_file = file_path.with_suffix('.csv').name
        fields = []
        for subject_id in filename_to_subject_id_mapping[tracklet_file]:
            fields.append({
                'type': 'field',
                'subject_id': subject_id,
                'set_num': None,
                'activity': None,
                'camera': file_path.suffixes[3][1:],
                'location': file_path.suffixes[2][1:],
                'time_start': datetime.strptime('.'.join(str(file_path.name).split('.')[:2]), '%Y-%m-%d.%H-%M-%S'),
                'time_stop': datetime.strptime('.'.join(np.array(str(file_path.name).split('.'))[[0, 2]]), '%Y-%m-%d.%H-%M-%S'),
                'file_path': file_path
            })
    elif 'brc2-enrollment' in str(file_path):
        if 'wholebody' in str(file_path):
            # fields = {
            #     'type': 'controlled',
            #     'activity': 'wholebody',
            #     'camera': 'flir',
            #     'location': 'wholebody_enrollment',
            #     'time_start': None,
            #     'subject_id': file_name[0].split('-')[0],
            #     'set_num': file_name[0].split('-')[1]
            # }
            raise ValueError
        if 'gait' in str(file_path):
            fields = {
                'type': 'controlled',
                'activity': file_name[2],
                'camera': file_path.suffixes[0][1:],
                'location': 'gait_enrollment',
                'time_start': None,
                'time_stop': None,
                'subject_id': file_name[0],
                'set_num': int(file_name[1][3:]),
                'file_path': file_path
            }
        fields = [fields]

    else:
        raise ValueError

    return fields


def intersection_over_union(points):
    """
    Calculates the intersection over union (IoU) of two rectangles defined by their top left and bottom right points.

    Args:
        p1: Top left corner coordinates of rectangle 1 (x, y)
        p2: Bottom right corner coordinates of rectangle 1 (x, y)
        p3: Top left corner coordinates of rectangle 2 (x, y)
        p4: Bottom right corner coordinates of rectangle 2 (x, y)

    Returns:
        The IoU value between 0 and 1.
    """
    p1, p2, p3, p4 = points
    # Unpack coordinates from points
    tl1_x, tl1_y = p1
    br1_x, br1_y = p2
    tl2_x, tl2_y = p3
    br2_x, br2_y = p4

    # Calculate intersection area
    tl_x = max(tl1_x, tl2_x)
    tl_y = max(tl1_y, tl2_y)
    br_x = min(br1_x, br2_x)
    br_y = min(br1_y, br2_y)
    intersection_area = (br_x - tl_x) * (br_y - tl_y)

    # Calculate areas of rectangles
    area_rect1 = (br1_x - tl1_x) * (br1_y - tl1_y)
    area_rect2 = (br2_x - tl2_x) * (br2_y - tl2_y)

    # Clip negative area to zero
    intersection_area = max(intersection_area, 0)

    # IoU calculation
    if (area_rect1 + area_rect2 - intersection_area) == 0:
        return 0
    iou = intersection_area / (area_rect1 + area_rect2 - intersection_area)

    return iou


def pad_to_square(image, pad_value=(0, 0, 0)):
    """
    Pads an image to have a 1:1 aspect ratio using the specified padding value.

    Args:
        image: The image to be padded (numpy array).
        pad_value: The RGB value for padding pixels (default: black (0, 0, 0)).

    Returns:
        The padded image (numpy array).
    """
    h, w = image.shape[:2]

    diff = abs(h - w)

    if h < w:
        left_pad = 0
        right_pad = 0
        top_pad = int(diff // 2)
        bottom_pad = int(diff - top_pad)
    else:
        top_pad = 0
        bottom_pad = 0
        left_pad = int(diff // 2)
        right_pad = int(diff - left_pad)

    # Create padding widths using NumPy
    padding = ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0))
    # Constant padding using slicing and broadcasting with pad_value
    padded_image = np.pad(image, padding, 'constant',
                          constant_values=(0, 0))

    h, w = padded_image.shape[:2]
    assert h == w
    return padded_image


def parse_dive_annotations(file_text, brc=1):

    file_text = '\n'.join(
        list(filter(lambda x: x[0] != '#', file_text.strip().splitlines())))
    if brc == 1:
        columns = ['track_id', 'timestamp', 'frame', 'TL_x', 'TL_y',
                   'BR_x', 'BR_y', 'confidence', 'null1', 'class', 'null2']
    elif brc == 2:
        columns = ['track_id', 'video_name', 'frame', 'TL_x', 'TL_y',
                   'BR_x', 'BR_y', 'confidence', 'null1', 'class', 'null2', 'null3']
    else:
        raise NotImplementedError
    annotations = pd.read_csv(
        StringIO(','.join(columns)+'\n'+file_text), delimiter=',')

    int_columns = ['frame', 'TL_x', 'TL_y', 'BR_x', 'BR_y']
    for column in int_columns:
        annotations[column] = annotations[column].astype(int)
    return annotations.set_index('frame')
