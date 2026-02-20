---
description: Allocate an interactive compute node with GPU for experiments
---

# Get Compute Node

Request an interactive compute node with H100 GPU for running experiments.

// turbo

## Step 1: Request the node

```bash
salloc --time=02:00:00 --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task= --mem=64G --account=rrg-ravanelm
```

> **Note**: This will allocate a node for 2 hours. Adjust `--time` as needed for longer experiments.

## Common time options:

- `--time=02:00:00` - 2 hours (quick tests)
- `--time=06:00:00` - 6 hours (medium runs)
- `--time=12:00:00` - 12 hours (full dataset runs)
