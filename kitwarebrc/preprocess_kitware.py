import os
import cv2
import numpy as np
from pathlib import Path
from kitbrc.brc import Dataset
from kitbrc.annotations.bounding_boxes import Tracklets

def main():
    # Paths to the necessary files
    video_manifest_path = '/blue/sarkar.sudeep/caio.dasilva/datasets/brc2-field/video_manifest.txt'  # Path to the video manifest file
    clip_id_to_filename_path = '/blue/sarkar.sudeep/mauricio.segundo/KITBRC2-new/KITBRC2/docs/KITBRC2-clipID-to-filename.txt'  # Path to the clip ID to filename mapping file
    subject_id_to_clip_id_path = '/blue/sarkar.sudeep/mauricio.segundo/KITBRC2-new/KITBRC2/docs/KITBRC2-subjectID-to-clipID.txt'  # Path to the subject ID to clip ID mapping file
    camera_bbox_bounds_path = 'camera_bbox_bounds.json'  # Path to the camera bounding box bounds file (optional)


    # Initialize the Dataset
    dataset = Dataset(
        video_manifest=video_manifest_path,
        camera_bbox_bounds=camera_bbox_bounds_path,
        brc=2,  # Set to 1 or 2 depending on your BRC version
        clip_id_to_filename=clip_id_to_filename_path,
        subject_id_to_clip_id=subject_id_to_clip_id_path
    )

    # Select one video for testing
    test_video = dataset.videos.iloc[0]
    
    # Print the keys/column names of the test_video DataFrame
    print("\nKeys for test_video DataFrame:", list(test_video.keys()))
    print(test_video['file_path'])

    print(test_video['camera'])



    video_path = test_video['file_path']



    



    # Create a Tracklets object
    tracklets = test_video['tracklets']
    
    # Filter the tracklets
    filtered_annotations = tracklets.filter()

    if filtered_annotations is None:
        print("No annotations found for the selected video.")
        return


    # Frame extraction and cropping
    output_dir = Path(f'output/{test_video["subject_id"]}/{test_video["camera"]}/frames')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0

   
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_count}")
            break

        # Extract tracklet for the current frame
        if frame_count in filtered_annotations.index:
            tracklet = filtered_annotations.loc[frame_count]
            x1, y1, x2, y2 = int(tracklet['TL_x']), int(tracklet['TL_y']), int(tracklet['BR_x']), int(tracklet['BR_y'])

            # Calculate the width and height of the bounding box
            width = x2 - x1
            height = y2 - y1
            print(f"Frame {frame_count}: Bounding Box Width = {width}, Height = {height}")

            # Calculate the center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Ensure the bounding box is at least 512x512
            new_width = max(512, width)
            new_height = max(512, height)

            # Calculate new bounding box coordinates
            x1 = center_x - new_width // 2
            x2 = center_x + new_width // 2
            y1 = center_y - new_height // 2
            y2 = center_y + new_height // 2

            # Ensure the coordinates are within the frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # Crop the frame based on the new bounding box
            cropped_frame = frame[y1:y2, x1:x2]

            # Create a blank 512x512 image
            output_frame = np.zeros((512, 512, 3), dtype=np.uint8)

            # Get the dimensions of the cropped frame
            cropped_h, cropped_w = cropped_frame.shape[:2]

            # Determine the top-left corner where the cropped frame will be placed
            start_y = (512 - cropped_h) // 2
            start_x = (512 - cropped_w) // 2

            # Place the cropped frame onto the blank image
            output_frame[start_y:start_y+cropped_h, start_x:start_x+cropped_w] = cropped_frame

            # Save the resized frame
            frame_output_path = output_dir / f'frame_{frame_count:04d}.png'
            cv2.imwrite(str(frame_output_path), output_frame)
        else:
            print(f"Frame {frame_count}: No bounding box found")

        frame_count += 1

    cap.release()

if __name__ == "__main__":
    main()