---
description: Resource monitoring template for sbatch scripts
---

# Resource Monitoring Skill

This skill provides a reusable resource monitoring snippet for sbatch scripts.

## Usage

Include this block in your sbatch scripts **after** environment setup and **before** the main Python command.

## Template

```bash
#==================================================================
# Resource Monitoring (runs in background every 20 minutes)
#==================================================================
while sleep 1200; do
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║              RESOURCE USAGE at $(date +%Y-%m-%d\ %H:%M:%S)              ║"
  echo "╠══════════════════════════════════════════════════════════════════╣"
  echo "║ MEMORY (RAM):"
  free -h | awk 'NR==2 {printf "║   Total: %s | Used: %s | Available: %s\n", $2, $3, $7}'
  echo "║"
  echo "║ CPU:"
  echo "║   Cores Allocated: ${SLURM_CPUS_PER_TASK:-N/A}"
  CPU_USAGE=$(ps -u $USER -o %cpu= | awk '{s+=$1} END {printf "%.1f", s}')
  echo "║   Total CPU Usage: ${CPU_USAGE}% (across all your processes)"
  echo "║"
  echo "║ GPU:"
  GPU_INFO=$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
  if [ -n "$GPU_INFO" ]; then
    echo "$GPU_INFO" | while IFS=',' read -r NAME UTIL MEM_USED MEM_TOTAL TEMP; do
      echo "║   Model: $NAME"
      echo "║   GPU Utilization: ${UTIL}% (how busy the GPU cores are)"
      echo "║   Memory Used: ${MEM_USED} MiB / ${MEM_TOTAL} MiB"
      echo "║   Temperature: ${TEMP}°C"
    done
  else
    echo "║   No GPU available"
  fi
  echo "╚══════════════════════════════════════════════════════════════════╝"
  echo ""
done &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT
```

## Example Output

```
╔══════════════════════════════════════════════════════════════════╗
║              RESOURCE USAGE at 2026-02-06 08:30:00              ║
╠══════════════════════════════════════════════════════════════════╣
║ MEMORY (RAM):
║   Total: 64G | Used: 12G | Available: 48G
║
║ CPU:
║   Cores Allocated: 8
║   Total CPU Usage: 125.3% (across all your processes)
║
║ GPU:
║   Model: NVIDIA H100 80GB HBM3
║   GPU Utilization: 45% (how busy the GPU cores are)
║   Memory Used: 12800 MiB / 40960 MiB
║   Temperature: 42°C
╚══════════════════════════════════════════════════════════════════╝
```

## Understanding the Metrics

| Metric                       | What it means                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------- |
| **RAM Total/Used/Available** | System memory allocation                                                        |
| **Cores Allocated**          | Number of CPU cores SLURM gave you                                              |
| **Total CPU Usage %**        | Sum of CPU% across all your processes (can exceed 100% if using multiple cores) |
| **GPU Utilization %**        | How busy the GPU compute cores are (100% = fully utilized)                      |
| **GPU Memory Used/Total**    | VRAM usage for model weights + activations                                      |
| **Temperature**              | GPU temperature in Celsius (typically <80°C is safe)                            |
