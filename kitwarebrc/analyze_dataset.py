import os
import collections
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def collect_dataset_info(root_dir):
    """
    Traverses the dataset directory and collects structural statistics.

    Args:
        root_dir (str): Root directory of the dataset.

    Returns:
        dict: Nested dataset statistics.
    """
    stats = {
        'subjects': {},
        'camera_frame_counts': collections.Counter(),
        'camera_subject_counts': collections.Counter(),
        'global_frame_count': 0,
        'global_subject_count': 0,
        'global_camera_count': 0,
        'camera_set': set(),
    }

    for subject in os.listdir(root_dir):
        subject_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subject_path):
            continue
        stats['global_subject_count'] += 1
        subject_cameras = {}
        subject_frame_count = 0
        for camera in os.listdir(subject_path):
            camera_path = os.path.join(subject_path, camera)
            if not os.path.isdir(camera_path):
                continue
            stats['camera_set'].add(camera)
            camera_path = os.path.join(camera_path, "frames")
            frames = [f for f in os.listdir(camera_path) if f.lower().endswith('.png')]
            n_frames = len(frames)
            subject_cameras[camera] = n_frames
            subject_frame_count += n_frames
            stats['camera_frame_counts'][camera] += n_frames
            if n_frames > 0:
                stats['camera_subject_counts'][camera] += 1

        stats['subjects'][subject] = {
            'cameras': subject_cameras,
            'frame_count': subject_frame_count,
            'n_cameras': len(subject_cameras)
        }
        stats['global_frame_count'] += subject_frame_count

    stats['global_camera_count'] = len(stats['camera_set'])
    return stats

def analyze_and_report(stats, show_plots=True, save_prefix=None):
    """
    Prints and plots distribution information from dataset statistics.

    Args:
        stats (dict): Output from collect_dataset_info.
        show_plots (bool): Whether to show the plots interactively.
        save_prefix (str|None): If set, save plots to files with this prefix.
    """
    print(f"Total subjects: {stats['global_subject_count']}")
    print(f"Total cameras: {stats['global_camera_count']}")
    print(f"Total frames: {stats['global_frame_count']}")
    print(f"Cameras: {sorted(stats['camera_set'])}")

    # Cameras per subject
    cameras_per_subject = [v['n_cameras'] for v in stats['subjects'].values()]
    print(f"\nCameras per subject distribution: min={min(cameras_per_subject)}, "
          f"max={max(cameras_per_subject)}, mean={sum(cameras_per_subject)/len(cameras_per_subject):.2f}")

    # Frames per subject
    frames_per_subject = [v['frame_count'] for v in stats['subjects'].values()]
    print(f"Frames per subject: min={min(frames_per_subject)}, "
          f"max={max(frames_per_subject)}, mean={sum(frames_per_subject)/len(frames_per_subject):.2f}")

    # Most common camera (by number of subjects using it)
    most_common_camera, n_subjects = stats['camera_subject_counts'].most_common(1)[0]
    print(f"\nMost common camera (by number of subjects): {most_common_camera} "
          f"used by {n_subjects} subjects")

    # Camera with most frames
    camera_most_frames, n_frames = stats['camera_frame_counts'].most_common(1)[0]
    print(f"Camera with highest number of frames: {camera_most_frames} ({n_frames} frames)")

    # Visualization setup
    subject_ids = []
    cameras = []
    frame_counts = []
    for subject, v in stats['subjects'].items():
        for camera, count in v['cameras'].items():
            subject_ids.append(subject)
            cameras.append(camera)
            frame_counts.append(count)
    df = pd.DataFrame({'subject': subject_ids, 'camera': cameras, 'frames': frame_counts})

    # Pivot table for per-camera stats
    pivot = df.pivot(index='subject', columns='camera', values='frames').fillna(0)

    # Per-camera frame/subject/min stats
    print("\nCamera stats (subjects, total frames, minimum frames per subject):")
    for cam in sorted(stats['camera_set']):
        n_subjects_cam = stats['camera_subject_counts'][cam]
        n_frames_cam = stats['camera_frame_counts'][cam]
        if cam in pivot.columns:
            nonzero = pivot[cam][pivot[cam] > 0]
            if not nonzero.empty:
                min_frames = int(nonzero.min())
            else:
                min_frames = "no subjects with frames"
        else:
            min_frames = "no subjects with frames"
        print(f"  {cam}: {n_subjects_cam} subjects, {n_frames_cam} frames, min frames/subject: {min_frames}")

    # --- Recommendation: Camera present in most subjects and with high average frames per subject ---
    camera_subjects = stats['camera_subject_counts']
    camera_frames = stats['camera_frame_counts']
    # Exclude 'flir' from all recommendations
    camera_set_recommend = set(stats['camera_set']) - {'flir'}
    avg_frames_per_subj = {}
    for cam in camera_set_recommend:
        n_subjects = camera_subjects[cam]
        total_frames = camera_frames[cam]
        avg_frames = total_frames / n_subjects if n_subjects > 0 else 0
        avg_frames_per_subj[cam] = avg_frames

    # Find the camera with the highest product of normalized #subjects and normalized avg frames
    if avg_frames_per_subj:
        max_n_subjects = max([camera_subjects[cam] for cam in camera_set_recommend])
        max_avg_frames = max(avg_frames_per_subj.values())
        camera_score = {}
        for cam in camera_set_recommend:
            # Normalize both metrics and multiply
            score = (camera_subjects[cam]/max_n_subjects if max_n_subjects > 0 else 0) * \
                    (avg_frames_per_subj[cam]/max_avg_frames if max_avg_frames > 0 else 0)
            camera_score[cam] = score
        best_balanced_camera = max(camera_score, key=camera_score.get)
        print(f"\nRecommended camera present in many subjects and with high frames per subject (excluding 'flir'):")
        print(f"  - {best_balanced_camera}: present in {camera_subjects[best_balanced_camera]} subjects, "
              f"avg frames per subject: {avg_frames_per_subj[best_balanced_camera]:.1f}")
    else:
        best_balanced_camera = None
        print("\nNo recommendation possible (no cameras except 'flir').")

    # Plot: Histogram of cameras per subject
    plt.figure(figsize=(8, 4))
    sns.histplot(list(cameras_per_subject), bins=range(1, max(cameras_per_subject)+2))
    plt.title("Distribution of Cameras per Subject")
    plt.xlabel("Number of Cameras")
    plt.ylabel("Number of Subjects")
    if save_prefix:
        plt.savefig(f"{save_prefix}_cameras_per_subject.png")
    if show_plots:
        plt.show()

    # Plot: Histogram of frames per subject
    plt.figure(figsize=(8, 4))
    sns.histplot(list(frames_per_subject), bins=20)
    plt.title("Distribution of Frames per Subject")
    plt.xlabel("Number of Frames")
    plt.ylabel("Number of Subjects")
    if save_prefix:
        plt.savefig(f"{save_prefix}_frames_per_subject.png")
    if show_plots:
        plt.show()

    # Plot: Boxplot of frames per subject per camera
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="camera", y="frames")
    plt.title("Frames per Subject per Camera")
    plt.xlabel("Camera")
    plt.ylabel("Frames per Subject")
    if save_prefix:
        plt.savefig(f"{save_prefix}_boxplot_frames_camera.png")
    if show_plots:
        plt.show()

    # Plot: Heatmap (subjects vs cameras, number of frames)
    plt.figure(figsize=(min(20, 3 + stats['global_camera_count']), max(6, 0.2*stats['global_subject_count'])))
    sns.heatmap(pivot, cmap="YlGnBu", annot=False, cbar=True)
    plt.title("Frames per Subject per Camera (Heatmap)")
    plt.xlabel("Camera")
    plt.ylabel("Subject")
    if save_prefix:
        plt.savefig(f"{save_prefix}_heatmap_subject_camera.png")
    if show_plots:
        plt.show()

    # Recommendations for balanced splits
    min_cams = min(cameras_per_subject)
    balanced_cameras = [cam for cam in camera_set_recommend
                        if camera_subjects[cam] == stats['global_subject_count']]
    print("\nRecommendation for balanced split (excluding 'flir'):")
    print(f"  - Only use subjects with at least {min_cams} cameras (all subjects have at least this).")
    if balanced_cameras:
        print(f"  - Use cameras present for all subjects: {balanced_cameras}")
    else:
        # If no camera is present for all subjects, present the one present in most subjects
        if camera_set_recommend:
            most_common_camera = max(camera_set_recommend, key=lambda cam: camera_subjects[cam])
            most_subjects = camera_subjects[most_common_camera]
            print(f"  - No camera is present for all subjects (excluding 'flir').")
            print(f"  - The camera present in the most subjects is: '{most_common_camera}', present in {most_subjects} subjects.")
        else:
            most_common_camera = None
            print("  - No camera to recommend (all cameras are 'flir').")

    # Minimum frames for the most balanced camera
    if best_balanced_camera is not None:
        camera_col = pivot[best_balanced_camera]
        nonzero_frames = camera_col[camera_col > 0]
        if not nonzero_frames.empty:
            min_frames_balanced = int(nonzero_frames.min())
            print(f"  - For '{best_balanced_camera}', subsample to at least {min_frames_balanced} frames per subject.")
        else:
            print(f"  - For '{best_balanced_camera}', no subjects have frames (unexpected).")

    # Minimum frames for the most common camera (if different)
    if camera_set_recommend and ('most_common_camera' in locals()) and (most_common_camera is not None):
        camera_col_common = pivot[most_common_camera]
        nonzero_frames_common = camera_col_common[camera_col_common > 0]
        if not nonzero_frames_common.empty:
            min_frames_common = int(nonzero_frames_common.min())
            print(f"  - For '{most_common_camera}', subsample to at least {min_frames_common} frames per subject.")
        else:
            print(f"  - For '{most_common_camera}', no subjects have frames (unexpected).")

    print("  - You may want to subsample to these minimums to ensure perfect balance for your chosen camera.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze dataset distribution for balancing splits.")
    parser.add_argument("root", help="Root directory of the dataset")
    parser.add_argument("--no-show", action="store_true", help="Do not show plots interactively")
    parser.add_argument("--save-prefix", default=None, help="If set, save plots with this prefix")
    args = parser.parse_args()

    stats = collect_dataset_info(args.root)
    analyze_and_report(stats, show_plots=not args.no_show, save_prefix=args.save_prefix)