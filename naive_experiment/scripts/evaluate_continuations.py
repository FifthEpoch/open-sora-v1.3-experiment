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


def extract_frames_from_video(video_path, start_frame=22, num_frames=23):
    """Extract frames from video starting at start_frame (frames 23-45)."""
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
    import sys
    import torch
    import torch.nn.functional as F
    
    metrics = {}
    
    # Ensure same number of frames (take minimum)
    min_frames = min(len(gt_frames), len(generated_frames))
    gt_frames = gt_frames[:min_frames]
    generated_frames = generated_frames[:min_frames]
    
    # Convert to torch tensors (N, H, W, C) -> (N, C, H, W)
    gt_tensor = torch.from_numpy(gt_frames).permute(0, 3, 1, 2).float() / 255.0
    gen_tensor = torch.from_numpy(generated_frames).permute(0, 3, 1, 2).float() / 255.0
    
    # Determine device (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Debug: Print shapes
    print(f"  [EVAL] GT tensor shape: {gt_tensor.shape}, Generated tensor shape: {gen_tensor.shape}", file=sys.stderr)
    
    # Resize generated frames to match GT resolution if needed
    if gt_tensor.shape[2:] != gen_tensor.shape[2:]:  # Compare H, W
        print(f"  [EVAL] Resolution mismatch: GT {gt_tensor.shape[2:]} vs Generated {gen_tensor.shape[2:]}", file=sys.stderr)
        print(f"  [EVAL] Resizing generated frames to match GT resolution...", file=sys.stderr)
        gen_tensor = F.interpolate(gen_tensor, size=gt_tensor.shape[2:], mode='bilinear', align_corners=False)
        print(f"  [EVAL] After resize: Generated tensor shape: {gen_tensor.shape}", file=sys.stderr)
    
    # Calculate PSNR (per frame, then average)
    psnr_values = []
    for i in range(min_frames):
        psnr_result = calculate_psnr(
            gt_tensor[i:i+1].unsqueeze(0),
            gen_tensor[i:i+1].unsqueeze(0)
        )
        # calculate_psnr returns a dict with 'value' key containing per-frame PSNRs
        if isinstance(psnr_result, dict) and 'value' in psnr_result:
            # Extract the PSNR value for frame 0 (we're passing one frame at a time)
            psnr_val = psnr_result['value'][0] if isinstance(psnr_result['value'], dict) else list(psnr_result['value'].values())[0]
        else:
            psnr_val = psnr_result
        psnr_values.append(float(psnr_val))
    metrics['psnr'] = np.mean(psnr_values)
    metrics['psnr_per_frame'] = psnr_values
    
    # Calculate SSIM (per frame, then average)
    ssim_values = []
    for i in range(min_frames):
        ssim_result = calculate_ssim(
            gt_tensor[i:i+1].unsqueeze(0),
            gen_tensor[i:i+1].unsqueeze(0)
        )
        # calculate_ssim returns a dict with 'value' key containing per-frame SSIMs
        if isinstance(ssim_result, dict) and 'value' in ssim_result:
            # Extract the SSIM value for frame 0 (we're passing one frame at a time)
            ssim_val = ssim_result['value'][0] if isinstance(ssim_result['value'], dict) else list(ssim_result['value'].values())[0]
        else:
            ssim_val = ssim_result
        ssim_values.append(float(ssim_val))
    metrics['ssim'] = np.mean(ssim_values)
    metrics['ssim_per_frame'] = ssim_values
    
    # Calculate LPIPS (per frame, then average)
    lpips_values = []
    for i in range(min_frames):
        lpips_result = calculate_lpips(
            gt_tensor[i:i+1].unsqueeze(0),
            gen_tensor[i:i+1].unsqueeze(0),
            device
        )
        # calculate_lpips returns a dict with 'value' key containing per-frame LPIPS
        if isinstance(lpips_result, dict) and 'value' in lpips_result:
            # Extract the LPIPS value for frame 0 (we're passing one frame at a time)
            lpips_val = lpips_result['value'][0] if isinstance(lpips_result['value'], dict) else list(lpips_result['value'].values())[0]
        else:
            lpips_val = lpips_result
        lpips_values.append(float(lpips_val))
    metrics['lpips'] = np.mean(lpips_values)
    metrics['lpips_per_frame'] = lpips_values
    
    return metrics


def main():
    import sys
    
    parser = argparse.ArgumentParser(description="Evaluate video continuations")
    parser.add_argument("--original-videos", type=str, required=True, help="Directory containing original videos")
    parser.add_argument("--baseline-outputs", type=str, required=True, help="Directory containing baseline outputs (O_b)")
    parser.add_argument("--finetuned-outputs", type=str, required=True, help="Directory containing fine-tuned outputs (O_f)")
    parser.add_argument("--manifest", type=str, required=True, help="CSV manifest with video mappings")
    parser.add_argument("--condition-frames", type=int, default=22, help="Number of conditioning frames")
    parser.add_argument("--output-json", type=str, default="metrics.json", help="Output JSON file for metrics")
    
    args = parser.parse_args()
    
    # Debug: Print to stderr so it shows up in SLURM .err file
    print(f"[EVALUATION] Loading manifest from: {args.manifest}", file=sys.stderr)
    
    # Load manifest
    manifest = pd.read_csv(args.manifest)
    print(f"[EVALUATION] Loaded manifest with {len(manifest)} rows", file=sys.stderr)
    print(f"[EVALUATION] Manifest columns: {list(manifest.columns)}", file=sys.stderr)
    print(f"[EVALUATION] First few rows:\n{manifest.head()}", file=sys.stderr)
    
    results = []
    
    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Evaluating"):
        import sys
        video_idx = row.get('video_idx', idx)
        original_path = row['original_path']
        baseline_output_raw = row.get('baseline_output', None)
        finetuned_output_raw = row.get('finetuned_output', None)
        
        print(f"\n[EVALUATION] Processing video {video_idx} (row {idx}):", file=sys.stderr)
        print(f"  Raw baseline_output type: {type(baseline_output_raw)}, value: {baseline_output_raw}", file=sys.stderr)
        print(f"  Raw finetuned_output type: {type(finetuned_output_raw)}, value: {str(finetuned_output_raw)[:200] if finetuned_output_raw else None}", file=sys.stderr)
        
        # Robustness: convert non-string/NaN paths to None to avoid os.path.exists(TypeError)
        if not isinstance(baseline_output_raw, str) or (isinstance(baseline_output_raw, str) and baseline_output_raw.strip() == ''):
            baseline_output = None
            print(f"  → baseline_output set to None (invalid type or empty)", file=sys.stderr)
        else:
            baseline_output = baseline_output_raw.strip()
        
        if not isinstance(finetuned_output_raw, str) or (isinstance(finetuned_output_raw, str) and finetuned_output_raw.strip() == ''):
            finetuned_output = None
            print(f"  → finetuned_output set to None (invalid type or empty)", file=sys.stderr)
        else:
            finetuned_output = finetuned_output_raw.strip()
            # Extract last line if it contains multiple lines (from stdout pollution fix)
            if '\n' in finetuned_output:
                lines = finetuned_output.strip().split('\n')
                # Take last line that looks like a path
                for line in reversed(lines):
                    if '/' in line and line.endswith('.mp4'):
                        finetuned_output = line.strip()
                        print(f"  → Extracted path from multi-line stdout: {finetuned_output}", file=sys.stderr)
                        break
        
        # Make paths absolute
        if original_path and not os.path.isabs(original_path):
            original_path = os.path.join(args.original_videos, os.path.basename(original_path))
            print(f"  Made original_path absolute: {original_path}", file=sys.stderr)
        
        # Ensure baseline and finetuned paths are absolute
        if baseline_output and not os.path.isabs(baseline_output):
            # If relative, assume it's relative to baseline_outputs directory
            baseline_output = os.path.join(args.baseline_outputs, baseline_output)
            print(f"  Made baseline_output absolute: {baseline_output}", file=sys.stderr)
        
        if finetuned_output and not os.path.isabs(finetuned_output):
            # If relative, assume it's relative to finetuned_outputs directory
            finetuned_output = os.path.join(args.finetuned_outputs, finetuned_output)
            print(f"  Made finetuned_output absolute: {finetuned_output}", file=sys.stderr)
        
        print(f"  Final paths:", file=sys.stderr)
        print(f"    original_path exists: {os.path.exists(original_path) if original_path else 'N/A'} - {original_path}", file=sys.stderr)
        print(f"    baseline_output exists: {os.path.exists(baseline_output) if baseline_output else 'N/A'} - {baseline_output}", file=sys.stderr)
        print(f"    finetuned_output exists: {os.path.exists(finetuned_output) if finetuned_output else 'N/A'} - {finetuned_output}", file=sys.stderr)
        
        result = {
            'video_idx': video_idx,
            'original_path': original_path,
        }
        
        # Extract GT frames - we'll determine how many based on what the generated video has
        # First, load one of the generated videos to see how many frames it has
        total_generated_frames = None
        if baseline_output and os.path.exists(baseline_output):
            try:
                temp_frames = load_generated_video(baseline_output)
                if temp_frames is not None:
                    total_generated_frames = len(temp_frames)
            except:
                pass
        
        if total_generated_frames is None and finetuned_output and os.path.exists(finetuned_output):
            try:
                temp_frames = load_generated_video(finetuned_output)
                if temp_frames is not None:
                    total_generated_frames = len(temp_frames)
            except:
                pass
        
        if total_generated_frames is None:
            result['error'] = "Could not determine generated video frame count"
            results.append(result)
            continue
        
        # Calculate expected continuation frames
        expected_continuation_frames = total_generated_frames - args.condition_frames
        print(f"  [GT] Detected {total_generated_frames} total frames, expecting {expected_continuation_frames} continuation frames", file=sys.stderr)
        
        # Extract GT frames (from condition_frames onwards)
        try:
            gt_frames = extract_frames_from_video(
                original_path,
                start_frame=args.condition_frames,
                num_frames=expected_continuation_frames
            )
            
            if gt_frames is None or len(gt_frames) == 0:
                result['error'] = "Could not extract GT frames"
                results.append(result)
                continue
            print(f"  [GT] Extracted {len(gt_frames)} GT frames for comparison", file=sys.stderr)
        except Exception as e:
            result['error'] = f"Error extracting GT: {str(e)}"
            results.append(result)
            continue
        
        # Evaluate baseline
        result['baseline_path'] = baseline_output
        result['baseline_path_exists'] = bool(baseline_output and os.path.exists(baseline_output))
        if baseline_output and os.path.exists(baseline_output):
            try:
                print(f"  [BASELINE] Loading video from: {baseline_output}", file=sys.stderr)
                baseline_frames = load_generated_video(baseline_output)
                if baseline_frames is not None:
                    print(f"  [BASELINE] Loaded {len(baseline_frames)} frames", file=sys.stderr)
                    # Extract continuation portion (skip conditioning frames)
                    baseline_continuation = baseline_frames[args.condition_frames:]
                    print(f"  [BASELINE] Continuation has {len(baseline_continuation)} frames (skipping first {args.condition_frames})", file=sys.stderr)
                    baseline_metrics = evaluate_pair(gt_frames, baseline_continuation)
                    print(f"  [BASELINE] Metrics computed: {baseline_metrics}", file=sys.stderr)
                    result['baseline'] = baseline_metrics
                else:
                    print(f"  [BASELINE] WARNING: load_generated_video returned None", file=sys.stderr)
                    result['baseline_error'] = "load_generated_video returned None"
            except Exception as e:
                import traceback
                print(f"  [BASELINE] ERROR: {str(e)}", file=sys.stderr)
                print(f"  [BASELINE] Traceback:\n{traceback.format_exc()}", file=sys.stderr)
                result['baseline_error'] = str(e)
        elif baseline_output:
            print(f"  [BASELINE] ERROR: File not found: {baseline_output}", file=sys.stderr)
            result['baseline_error'] = f"Baseline output not found: {baseline_output}"
        else:
            print(f"  [BASELINE] SKIP: No baseline_output path provided", file=sys.stderr)
        
        # Evaluate fine-tuned
        result['finetuned_path'] = finetuned_output
        result['finetuned_path_exists'] = bool(finetuned_output and os.path.exists(finetuned_output))
        if finetuned_output and os.path.exists(finetuned_output):
            try:
                print(f"  [FINETUNED] Loading video from: {finetuned_output}", file=sys.stderr)
                finetuned_frames = load_generated_video(finetuned_output)
                if finetuned_frames is not None:
                    print(f"  [FINETUNED] Loaded {len(finetuned_frames)} frames", file=sys.stderr)
                    # Extract continuation portion
                    finetuned_continuation = finetuned_frames[args.condition_frames:]
                    print(f"  [FINETUNED] Continuation has {len(finetuned_continuation)} frames (skipping first {args.condition_frames})", file=sys.stderr)
                    finetuned_metrics = evaluate_pair(gt_frames, finetuned_continuation)
                    print(f"  [FINETUNED] Metrics computed: {finetuned_metrics}", file=sys.stderr)
                    result['finetuned'] = finetuned_metrics
                else:
                    print(f"  [FINETUNED] WARNING: load_generated_video returned None", file=sys.stderr)
                    result['finetuned_error'] = "load_generated_video returned None"
            except Exception as e:
                import traceback
                print(f"  [FINETUNED] ERROR: {str(e)}", file=sys.stderr)
                print(f"  [FINETUNED] Traceback:\n{traceback.format_exc()}", file=sys.stderr)
                result['finetuned_error'] = str(e)
        elif finetuned_output:
            print(f"  [FINETUNED] ERROR: File not found: {finetuned_output}", file=sys.stderr)
            result['finetuned_error'] = f"Finetuned output not found: {finetuned_output}"
        else:
            print(f"  [FINETUNED] SKIP: No finetuned_output path provided", file=sys.stderr)
        
        results.append(result)
    
    # Save results
    print(f"\n[EVALUATION] Completed evaluation of {len(results)} videos", file=sys.stderr)
    print(f"[EVALUATION] Saving results to: {args.output_json}", file=sys.stderr)
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

