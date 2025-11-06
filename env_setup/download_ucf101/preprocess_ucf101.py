#!/usr/bin/env python3
"""
Preprocess UCF-101 videos for Open-Sora v1.3 training.

This script:
1. Center-crops videos to 640×480 (upscales from 320×240)
2. Resamples to 24 fps
3. Crops to uniform 45 frames
4. Skips videos shorter than 45 frames after resampling
5. Generates metadata CSV for training
"""

import argparse
import os
import shutil
from pathlib import Path
import av
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2


def center_crop_resize(frame, target_height=480, target_width=640):
    """
    Center crop and resize frame to target dimensions.
    UCF-101 is 320×240, we upscale to 640×480.
    """
    # Safety check: ensure frame is a numpy array
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Frame must be numpy array, got {type(frame)}")
    
    # Ensure frame has correct shape (H, W, C)
    if len(frame.shape) != 3:
        raise ValueError(f"Frame must have 3 dimensions (H, W, C), got shape {frame.shape}")
    
    # Ensure frame is contiguous and proper dtype for cv2
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    
    # Make a clean copy to ensure cv2 compatibility
    frame = np.copy(frame)
    
    h, w = frame.shape[:2]
    
    # Calculate aspect ratios
    target_aspect = target_width / target_height
    current_aspect = w / h
    
    # First, crop to target aspect ratio
    if current_aspect > target_aspect:
        # Video is wider, crop width
        new_w = int(h * target_aspect)
        start_x = (w - new_w) // 2
        frame = frame[:, start_x:start_x + new_w]
    elif current_aspect < target_aspect:
        # Video is taller, crop height
        new_h = int(w / target_aspect)
        start_y = (h - new_h) // 2
        frame = frame[start_y:start_y + new_h, :]
    
    # Ensure cropped frame is also contiguous
    frame = np.ascontiguousarray(frame)
    
    # Then resize to target dimensions
    # Workaround for cv2.resize bug with PyAV frames: convert to BGR and back
    try:
        # Try direct resize first
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    except Exception as e:
        # If that fails, try converting color space as workaround
        # This forces cv2 to make a proper internal copy
        try:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_bgr = cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e2:
            # Last resort: use PIL
            from PIL import Image
            img_pil = Image.fromarray(frame)
            img_pil = img_pil.resize((target_width, target_height), Image.LANCZOS)
            frame = np.array(img_pil)
    
    return frame


def resample_video(frames, original_fps, target_fps=24):
    """
    Resample video to target fps using linear interpolation.
    """
    if abs(original_fps - target_fps) < 0.1:
        return frames
    
    num_frames = len(frames)
    original_duration = num_frames / original_fps
    target_num_frames = int(original_duration * target_fps)
    
    if target_num_frames == 0:
        return frames
    
    # Create interpolation indices
    original_indices = np.arange(num_frames)
    target_indices = np.linspace(0, num_frames - 1, target_num_frames)
    
    # Interpolate frames
    resampled_frames = []
    for idx in target_indices:
        lower_idx = int(np.floor(idx))
        upper_idx = min(int(np.ceil(idx)), num_frames - 1)
        alpha = idx - lower_idx
        
        if lower_idx == upper_idx:
            frame = frames[lower_idx]
        else:
            # Linear interpolation
            frame = ((1 - alpha) * frames[lower_idx] + alpha * frames[upper_idx]).astype(np.uint8)
        
        resampled_frames.append(frame)
    
    return resampled_frames


def crop_to_n_frames(frames, n=45):
    """
    Uniformly crop video to exactly n frames.
    Uses uniform sampling across the video length.
    """
    if len(frames) < n:
        return None
    
    if len(frames) == n:
        return frames
    
    # Uniform sampling
    indices = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in indices]


def read_video(video_path):
    """Read video and return frames and fps."""
    try:
        container = av.open(str(video_path))
        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate)
        
        frames = []
        for frame in container.decode(video=0):
            try:
                # Convert frame to numpy array
                img = frame.to_ndarray(format='rgb24')
                
                # Verify it's actually a numpy array
                if not isinstance(img, np.ndarray):
                    print(f"Warning: Frame is not numpy array, type: {type(img)}")
                    continue
                
                # Verify shape is correct (H, W, C)
                if len(img.shape) != 3 or img.shape[2] != 3:
                    print(f"Warning: Frame has incorrect shape: {img.shape}")
                    continue
                
                # Ensure proper memory layout and contiguity for OpenCV
                # PyAV sometimes returns non-contiguous arrays that cv2 can't handle
                if not img.flags['C_CONTIGUOUS']:
                    img = np.ascontiguousarray(img)
                
                # Ensure uint8 dtype (required by OpenCV)
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                
                frames.append(img)
            except Exception as e:
                print(f"Warning: Failed to convert frame: {e}")
                continue
        
        container.close()
        
        if len(frames) == 0:
            print(f"Error: No valid frames extracted from {video_path}")
            return None, None
        
        return frames, fps
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        return None, None


def write_video(frames, output_path, fps=24):
    """Write frames to video file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    height, width = frames[0].shape[:2]
    
    container = av.open(str(output_path), mode='w')
    stream = container.add_stream('libx264', rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '18'}
    
    for frame in frames:
        av_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        for packet in stream.encode(av_frame):
            container.mux(packet)
    
    # Flush stream
    for packet in stream.encode():
        container.mux(packet)
    
    container.close()


def parse_ucf101_filename(filename):
    """
    Parse UCF-101 filename to extract class name.
    Format: v_ActionName_g##_c##.avi
    """
    if not filename.startswith('v_'):
        return None
    
    name_without_prefix = filename[2:]
    name_without_ext = name_without_prefix.rsplit('.', 1)[0]
    parts = name_without_ext.split('_')
    
    if len(parts) < 3:
        return None
    
    class_name = '_'.join(parts[:-2])
    return class_name


def process_video(video_path, output_base, target_fps=24, target_frames=45, target_height=480, target_width=640):
    """
    Process a single video:
    1. Read video
    2. Center crop and resize each frame
    3. Resample to target fps
    4. Crop to target frames
    5. Write to output
    
    Returns: (success, output_path, num_frames, caption) or (False, None, None, None)
    """
    try:
        # Read video
        frames, fps = read_video(video_path)
        if frames is None or len(frames) == 0:
            print(f"  DEBUG: read_video returned None or empty for {video_path.name}")
            return False, None, None, None
        
        print(f"  DEBUG: Loaded {len(frames)} frames at {fps:.2f} fps from {video_path.name}")
        
        # Center crop and resize - process each frame with error handling
        processed_frames = []
        for i, frame in enumerate(frames):
            try:
                # Double-check frame validity before processing
                if not isinstance(frame, np.ndarray):
                    print(f"  Warning: Frame {i} is not numpy array in {video_path.name}, skipping")
                    continue
                
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    print(f"  Warning: Frame {i} has invalid shape {frame.shape} in {video_path.name}, skipping")
                    continue
                
                # Debug first frame to see what's wrong
                if i == 0:
                    print(f"  DEBUG Frame 0: type={type(frame)}, dtype={frame.dtype}, shape={frame.shape}, "
                          f"contiguous={frame.flags['C_CONTIGUOUS']}, "
                          f"writeable={frame.flags['WRITEABLE']}")
                
                processed_frame = center_crop_resize(frame, target_height, target_width)
                processed_frames.append(processed_frame)
            except Exception as e:
                print(f"  Warning: Failed to process frame {i} in {video_path.name}: {e}")
                if i == 0:
                    import traceback
                    print(f"  Full traceback for frame 0:")
                    traceback.print_exc()
                continue
        
        if len(processed_frames) == 0:
            print(f"  Error: No frames could be processed for {video_path.name}")
            return False, None, None, None
        
        print(f"  DEBUG: After processing, have {len(processed_frames)} frames")
        
        # Resample to target fps
        resampled_frames = resample_video(processed_frames, fps, target_fps)
        print(f"  DEBUG: After resampling from {fps:.2f} to {target_fps} fps, have {len(resampled_frames)} frames")
        
        # Crop to target number of frames
        cropped_frames = crop_to_n_frames(resampled_frames, target_frames)
        if cropped_frames is None:
            print(f"  DEBUG: crop_to_n_frames returned None - video too short ({len(resampled_frames)} < {target_frames} frames required)")
            return False, None, None, None
        
        print(f"  DEBUG: After cropping to {target_frames} frames, ready to write")
        
        # Determine output path (maintain directory structure)
        relative_path = video_path.relative_to(video_path.parents[1])  # Relative to ucf101_org parent
        output_path = output_base / relative_path.parent / f"{video_path.stem}.mp4"
        
        # Write video
        write_video(cropped_frames, output_path, target_fps)
        
        # Get caption from filename or directory
        class_name = parse_ucf101_filename(video_path.name)
        if class_name is None:
            class_name = video_path.parent.name
        
        # Convert to readable caption
        caption = ''.join([' ' + c.lower() if c.isupper() else c for c in class_name]).strip()
        caption = caption.replace('_', ' ')
        
        return True, output_path, target_frames, caption
    
    except Exception as e:
        print(f"  Error processing video {video_path.name}: {e}")
        return False, None, None, None


def main():
    parser = argparse.ArgumentParser(description="Preprocess UCF-101 videos for Open-Sora v1.3")
    parser.add_argument("--input-dir", type=str, default="ucf101_org",
                       help="Input directory containing UCF-101 videos")
    parser.add_argument("--output-dir", type=str, default="ucf101_processed",
                       help="Output directory for preprocessed videos")
    parser.add_argument("--fps", type=int, default=24,
                       help="Target frame rate")
    parser.add_argument("--frames", type=int, default=45,
                       help="Target number of frames")
    parser.add_argument("--height", type=int, default=480,
                       help="Target height")
    parser.add_argument("--width", type=int, default=640,
                       help="Target width")
    parser.add_argument("--conditioning-frames", type=int, default=22,
                       help="Number of conditioning frames for metadata")
    parser.add_argument("--skip-cleanup", action="store_true",
                       help="Skip cleanup prompt (for non-interactive execution)")
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input_dir
    output_dir = script_dir / args.output_dir
    
    if not input_dir.exists():
        print(f"❌ ERROR: Input directory not found: {input_dir}")
        print("Please run download_ucf101.py first!")
        return
    
    print("=" * 70)
    print("UCF-101 Video Preprocessing")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target resolution: {args.width}×{args.height}")
    print(f"Target FPS: {args.fps}")
    print(f"Target frames: {args.frames}")
    print(f"Conditioning frames: {args.conditioning_frames}")
    print("=" * 70)
    
    # Find all videos
    video_files = list(input_dir.rglob("*.avi"))
    print(f"\nFound {len(video_files)} videos to process")
    
    if len(video_files) == 0:
        print("❌ No videos found! Check input directory.")
        return
    
    # Process videos
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    successful = 0
    skipped = 0
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        success, output_path, num_frames, caption = process_video(
            video_path, output_dir, args.fps, args.frames, args.height, args.width
        )
        
        if success:
            # Add to metadata
            metadata.append({
                'path': str(output_path.relative_to(script_dir)),
                'num_frames': num_frames,
                'conditioning_frames': args.conditioning_frames,
                'height': args.height,
                'width': args.width,
                'fps': args.fps,
                'text': caption
            })
            successful += 1
        else:
            skipped += 1
    
    # Save metadata CSV
    csv_path = script_dir / "ucf101_metadata.csv"
    df = pd.DataFrame(metadata)
    df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 70)
    print("✓ Preprocessing complete!")
    print("=" * 70)
    print(f"Successful: {successful} videos")
    print(f"Skipped: {skipped} videos (too short)")
    print(f"Output directory: {output_dir}")
    print(f"Metadata CSV: {csv_path}")
    
    # Optional cleanup
    if not args.skip_cleanup:
        print("\n" + "=" * 70)
        print("Disk Space Management")
        print("=" * 70)
        print("Original videos can be deleted to save disk space.")
        response = input("Delete original videos? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            print(f"\nDeleting {input_dir}...")
            shutil.rmtree(input_dir)
            print("✓ Original videos deleted!")
        else:
            print("Keeping original videos.")
    
    print("\n✓ All done! Ready for training.")


if __name__ == "__main__":
    main()

