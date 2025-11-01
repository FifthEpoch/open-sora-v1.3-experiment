#!/usr/bin/env python3
"""
Evaluate video continuations using PSNR, SSIM, and LPIPS metrics.

Compares generated continuations (O_b and O_f) against ground truth (GT).
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import evaluation functions from Open-Sora's eval/vae
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "eval" / "vae"))

try:
    from cal_psnr import calculate_psnr
    from cal_ssim import calculate_ssim
    from cal_lpips import calculate_lpips
except ImportError:
    print("Warning: Could not import evaluation functions from eval/vae")
    print("Make sure you're running from the Open-Sora root directory")


def extract_frames_from_video(video_path, start_frame=32, num_frames=13):
    """Extract frames from video starting at start_frame."""
    import av
    
    container = av.open(str(video_path))
    frames = []
    frame_idx = 0
    
    for frame in container.decode(video=0):
        if frame_idx >= start_frame and frame_idx < start_frame + num_frames:
            frames.append(frame.to_ndarray(format='rgb24'))
        elif frame_idx >= start_frame + num_frames:
            break
        frame_idx += 1
    
    container.close()
    return np.stack(frames) if frames else None


def load_generated_video(video_path):
    """Load all frames from a generated video."""
    import av
    
    container = av.open(str(video_path))
    frames = []
    
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format='rgb24'))
    
    container.close()
    return np.stack(frames) if frames else None


def evaluate_pair(gt_frames, generated_frames):
    """Evaluate a pair of videos (GT vs generated continuation)."""
    metrics = {}
    
    # Ensure same number of frames (take minimum)
    min_frames = min(len(gt_frames), len(generated_frames))
    gt_frames = gt_frames[:min_frames]
    generated_frames = generated_frames[:min_frames]
    
    # Convert to torch tensors (N, H, W, C) -> (N, C, H, W)
    import torch
    
    gt_tensor = torch.from_numpy(gt_frames).permute(0, 3, 1, 2).float() / 255.0
    gen_tensor = torch.from_numpy(generated_frames).permute(0, 3, 1, 2).float() / 255.0
    
    # Calculate PSNR (per frame, then average)
    psnr_values = []
    for i in range(min_frames):
        psnr = calculate_psnr(
            gt_tensor[i:i+1].unsqueeze(0),
            gen_tensor[i:i+1].unsqueeze(0)
        )
        psnr_values.append(psnr.item() if torch.is_tensor(psnr) else psnr)
    metrics['psnr'] = np.mean(psnr_values)
    metrics['psnr_per_frame'] = psnr_values
    
    # Calculate SSIM (per frame, then average)
    ssim_values = []
    for i in range(min_frames):
        ssim = calculate_ssim(
            gt_tensor[i:i+1].unsqueeze(0),
            gen_tensor[i:i+1].unsqueeze(0)
        )
        ssim_values.append(ssim.item() if torch.is_tensor(ssim) else ssim)
    metrics['ssim'] = np.mean(ssim_values)
    metrics['ssim_per_frame'] = ssim_values
    
    # Calculate LPIPS (per frame, then average)
    lpips_values = []
    for i in range(min_frames):
        lpips = calculate_lpips(
            gt_tensor[i:i+1].unsqueeze(0),
            gen_tensor[i:i+1].unsqueeze(0)
        )
        lpips_values.append(lpips.item() if torch.is_tensor(lpips) else lpips)
    metrics['lpips'] = np.mean(lpips_values)
    metrics['lpips_per_frame'] = lpips_values
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate video continuations")
    parser.add_argument("--original-videos", type=str, required=True, help="Directory containing original videos")
    parser.add_argument("--baseline-outputs", type=str, required=True, help="Directory containing baseline outputs (O_b)")
    parser.add_argument("--finetuned-outputs", type=str, required=True, help="Directory containing fine-tuned outputs (O_f)")
    parser.add_argument("--manifest", type=str, required=True, help="CSV manifest with video mappings")
    parser.add_argument("--condition-frames", type=int, default=8, help="Number of conditioning frames")
    parser.add_argument("--output-json", type=str, default="metrics.json", help="Output JSON file for metrics")
    
    args = parser.parse_args()
    
    # Load manifest
    manifest = pd.read_csv(args.manifest)
    
    results = []
    
    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Evaluating"):
        video_idx = row.get('video_idx', idx)
        original_path = row['original_path']
        baseline_output = row.get('baseline_output', None)
        finetuned_output = row.get('finetuned_output', None)
        
        # Make paths absolute
        if not os.path.isabs(original_path):
            original_path = os.path.join(args.original_videos, os.path.basename(original_path))
        
        result = {
            'video_idx': video_idx,
            'original_path': original_path,
        }
        
        # Extract GT frames (frames 32-44 from original video)
        try:
            gt_frames = extract_frames_from_video(
                original_path,
                start_frame=args.condition_frames,
                num_frames=45 - args.condition_frames
            )
            
            if gt_frames is None or len(gt_frames) == 0:
                result['error'] = "Could not extract GT frames"
                results.append(result)
                continue
        except Exception as e:
            result['error'] = f"Error extracting GT: {str(e)}"
            results.append(result)
            continue
        
        # Evaluate baseline
        if baseline_output and os.path.exists(baseline_output):
            try:
                baseline_frames = load_generated_video(baseline_output)
                if baseline_frames is not None:
                    # Extract continuation portion (skip conditioning frames)
                    baseline_continuation = baseline_frames[args.condition_frames:]
                    baseline_metrics = evaluate_pair(gt_frames, baseline_continuation)
                    result['baseline'] = baseline_metrics
            except Exception as e:
                result['baseline_error'] = str(e)
        
        # Evaluate fine-tuned
        if finetuned_output and os.path.exists(finetuned_output):
            try:
                finetuned_frames = load_generated_video(finetuned_output)
                if finetuned_frames is not None:
                    # Extract continuation portion
                    finetuned_continuation = finetuned_frames[args.condition_frames:]
                    finetuned_metrics = evaluate_pair(gt_frames, finetuned_continuation)
                    result['finetuned'] = finetuned_metrics
            except Exception as e:
                result['finetuned_error'] = str(e)
        
        results.append(result)
    
    # Save results
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Compute summary statistics
    baseline_psnr = [r.get('baseline', {}).get('psnr') for r in results if 'baseline' in r]
    baseline_ssim = [r.get('baseline', {}).get('ssim') for r in results if 'baseline' in r]
    baseline_lpips = [r.get('baseline', {}).get('lpips') for r in results if 'baseline' in r]
    
    finetuned_psnr = [r.get('finetuned', {}).get('psnr') for r in results if 'finetuned' in r]
    finetuned_ssim = [r.get('finetuned', {}).get('ssim') for r in results if 'finetuned' in r]
    finetuned_lpips = [r.get('finetuned', {}).get('lpips') for r in results if 'finetuned' in r]
    
    print("\n" + "="*70)
    print("Evaluation Summary")
    print("="*70)
    print(f"\nBaseline (O_b):")
    if baseline_psnr:
        print(f"  PSNR: {np.mean(baseline_psnr):.4f} ± {np.std(baseline_psnr):.4f}")
    if baseline_ssim:
        print(f"  SSIM: {np.mean(baseline_ssim):.4f} ± {np.std(baseline_ssim):.4f}")
    if baseline_lpips:
        print(f"  LPIPS: {np.mean(baseline_lpips):.4f} ± {np.std(baseline_lpips):.4f}")
    
    print(f"\nFine-tuned (O_f):")
    if finetuned_psnr:
        print(f"  PSNR: {np.mean(finetuned_psnr):.4f} ± {np.std(finetuned_psnr):.4f}")
    if finetuned_ssim:
        print(f"  SSIM: {np.mean(finetuned_ssim):.4f} ± {np.std(finetuned_ssim):.4f}")
    if finetuned_lpips:
        print(f"  LPIPS: {np.mean(finetuned_lpips):.4f} ± {np.std(finetuned_lpips):.4f}")
    
    print(f"\nImprovement:")
    if baseline_psnr and finetuned_psnr:
        psnr_improvement = np.mean(finetuned_psnr) - np.mean(baseline_psnr)
        print(f"  PSNR: {psnr_improvement:+.4f}")
    if baseline_ssim and finetuned_ssim:
        ssim_improvement = np.mean(finetuned_ssim) - np.mean(baseline_ssim)
        print(f"  SSIM: {ssim_improvement:+.4f}")
    if baseline_lpips and finetuned_lpips:
        lpips_improvement = np.mean(baseline_lpips) - np.mean(finetuned_lpips)  # Lower is better
        print(f"  LPIPS: {lpips_improvement:+.4f} (lower is better)")
    
    print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()

