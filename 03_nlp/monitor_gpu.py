#!/usr/bin/env python3
"""
GPU monitoring script for translation optimization.
Run alongside translation to see real-time GPU stats.
"""

import subprocess
import time
import sys
from datetime import datetime


def get_gpu_stats():
    """Get GPU utilization and memory usage."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )

        values = result.stdout.strip().split(',')
        return {
            'gpu_util': int(values[0].strip()),
            'mem_util': int(values[1].strip()),
            'mem_used': int(values[2].strip()),
            'mem_total': int(values[3].strip())
        }
    except Exception as e:
        return None


def format_size(mb):
    """Format memory size."""
    if mb < 1024:
        return f"{mb}MB"
    return f"{mb/1024:.1f}GB"


def main():
    print("GPU Monitoring (press Ctrl+C to stop)")
    print("=" * 80)

    try:
        max_util = 0
        max_mem = 0
        samples = []

        while True:
            stats = get_gpu_stats()

            if stats:
                timestamp = datetime.now().strftime("%H:%M:%S")
                gpu_util = stats['gpu_util']
                mem_util = stats['mem_util']
                mem_used = stats['mem_used']
                mem_total = stats['mem_total']
                mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0

                max_util = max(max_util, gpu_util)
                max_mem = max(max_mem, mem_pct)
                samples.append(gpu_util)

                # Progress bar for GPU utilization
                bar_length = 40
                filled = int(bar_length * gpu_util / 100)
                bar = '█' * filled + '░' * (bar_length - filled)

                print(f"\r[{timestamp}] GPU: {bar} {gpu_util:3d}% | "
                      f"VRAM: {format_size(mem_used)}/{format_size(mem_total)} ({mem_pct:.0f}%) | "
                      f"Peak: {max_util}%",
                      end='', flush=True)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        if samples:
            avg_util = sum(samples) / len(samples)
            print(f"Statistics:")
            print(f"  Average GPU utilization: {avg_util:.1f}%")
            print(f"  Peak GPU utilization: {max_util}%")
            print(f"  Peak VRAM usage: {max_mem:.1f}%")
            print(f"\nRecommendations:")
            if avg_util < 50:
                print("  ⚠ Low GPU utilization - consider increasing batch size")
            elif avg_util < 80:
                print("  ✓ Good GPU utilization - room for small increase")
            else:
                print("  ✓ Excellent GPU utilization!")
        print("=" * 80)


if __name__ == "__main__":
    main()
