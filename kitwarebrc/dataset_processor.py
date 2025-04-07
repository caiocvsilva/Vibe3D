import os
import sys
import shutil
import cv2
import numpy as np

# Removed face_recognition import since it is no longer used

# Threshold for clothing descriptor comparison (lower means more similar)
CLOTHES_DISTANCE_THRESHOLD = 0.1

def get_clothes_descriptor(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    h = image.shape[0]
    region = image[h//2:, :]  # use lower half for clothing
    hist = cv2.calcHist([region], [0, 1, 2], None, [8,8,8], [0,256,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def group_frames(frame_paths):
    groups = {}  # key: group id, value: list of (frame_path, descriptor)
    next_group = 1
    for path in frame_paths:
        descriptor = get_clothes_descriptor(path)
        if descriptor is None:
            print(f"Warning: No clothing descriptor found for {path}. Skipping.")
            continue
        assigned = False
        for group_id, members in groups.items():
            rep_descriptor = members[0][1]
            distance = np.linalg.norm(descriptor - rep_descriptor)
            if distance < CLOTHES_DISTANCE_THRESHOLD:
                groups[group_id].append((path, descriptor))
                assigned = True
                break
        if not assigned:
            groups[next_group] = [(path, descriptor)]
            next_group += 1
    return groups

def show_comparison(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    if img1 is None or img2 is None:
        print("Error reading images for comparison.")
        return None
    # Resize images to have the same height
    h = 400
    img1 = cv2.resize(img1, (int(img1.shape[1] * h/img1.shape[0]), h))
    img2 = cv2.resize(img2, (int(img2.shape[1] * h/img2.shape[0]), h))
    combined = np.hstack((img1, img2))
    cv2.imshow("Comparison (Left: Group 1, Right: Group 2)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_camera(camera_path):
    frames_dir = os.path.join(camera_path, "frames")
    if not os.path.exists(frames_dir):
        print(f"No frames directory in {camera_path}. Skipping.")
        return
    frame_files = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png")]
    if not frame_files:
        print(f"No PNG frames in {frames_dir}. Skipping.")
        return

    groups = group_frames(frame_files)
    if len(groups) <= 1:
        print(f"All frames in {frames_dir} appear to have the same clothes.")
        return
    elif len(groups) == 2:
        group_ids = sorted(groups.keys())
        rep1 = groups[group_ids[0]][0][0]
        rep2 = groups[group_ids[1]][0][0]
        print(f"Two different clothing types detected in {frames_dir}.")
        show_comparison(rep1, rep2)
        choice = input("Which group should be kept? Type 1 for left image or 2 for right image: ").strip()
        try:
            keep_group = group_ids[int(choice)-1]
        except (ValueError, IndexError):
            print("Invalid choice. Skipping deletion.")
            return
        for gid, items in groups.items():
            if gid != keep_group:
                for (fp, _) in items:
                    print(f"Deleting {fp}")
                    os.remove(fp)
    else:
        print(f"More than two clothing types detected in {frames_dir}. Manual intervention required.")
        # Alternatively, implement additional handling for >2 groups

def main(root_dir):
    if not os.path.isdir(root_dir):
        print(f"{root_dir} is not a directory")
        return

    # Iterate over subject subdirectories
    for subject in os.listdir(root_dir):
        subject_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subject_path):
            continue
        # Iterate over camera directories in each subject
        for camera in os.listdir(subject_path):
            camera_path = os.path.join(subject_path, camera)
            if not os.path.isdir(camera_path):
                continue
            if camera.lower() == "firf":
                print(f"Removing camera folder {camera_path}")
                shutil.rmtree(camera_path)
                continue
            process_camera(camera_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dataset_processor.py <root_dataset_directory>")
    else:
        main(sys.argv[1])
