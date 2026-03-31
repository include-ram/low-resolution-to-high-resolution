# Super Resolution Project - Multi-GPU Training Guide

This project contains four distributed training scripts for super-resolution using PyTorch:
1. `test_minimal_rrdb.py` (DDP)
2. `esrgan_fsdp.py` (FSDP)
3. `vqgan_transformer_sr_distributed_fixed.py` (DDP)
4. `vqgan_transformer_fsdp.py` (FSDP)

---

## Running the Code

###  1 GPU (Single GPU)

python -m torch.distributed.launch --nproc_per_node=1 --master_port=65432 your_script.py


For example:

python -m torch.distributed.launch --nproc_per_node=1 --master_port=65432 vqgan_transformer_sr_distributed_fixed.py


###2 GPUs (Distributed)

python -m torch.distributed.launch --nproc_per_node=2 --master_port=65432 your_script.py --world_size 2


Example:

python test_minimal_rrdb.py --world_size 2


## 4 GPUs (Distributed)

python -m torch.distributed.launch --nproc_per_node=4 --master_port=65432 your_script.py --world_size 4


---

## Notes on Configuration Changes for Multi-GPU Efficiency

When using more GPUs, **you should adjust these settings** in the `config` dictionary inside each script:

1. **`batch_size`**:
   - Increase per-GPU batch size only if memory permits.
   - Or keep constant and use gradient accumulation.

2. **`gradient_accumulation_steps`** (in ESRGAN FSDP):
   - This setting should scale with GPU count:
     ```python
     'gradient_accumulation_steps': {
         1: 2,   # 1 GPU
         2: 4,   # 2 GPUs
         4: 8    # 4 GPUs
     }
     ```

3. **`num_workers`**:
   - Should be scaled with the number of GPUs or CPU cores:
     ```python
     'num_workers': 4 * NUM_GPUS
     ```

4. **FSDP-specific (for `vqgan_transformer_fsdp.py` or `esrgan_fsdp.py`)**:
   - Use `--cpu_offload` and `--use_mixed_precision` for memory savings and speed.
   - Tune `--min_params_for_wrap` for auto-wrapping deeper layers.

---

## Examples of Best Config Tweaks for 2/4 GPUs

For `esrgan_fsdp.py`:
```python
'gradient_accumulation_steps': {
    1: 2,
    2: 4,
    4: 8
},
'batch_size': 4,
'num_workers': 4
```

For `vqgan_transformer_fsdp.py`:
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=65432 vqgan_transformer_fsdp.py --world_size 4 --use_mixed_precision --cpu_offload --min_params_for_wrap 100000
```

---

## Troubleshooting
- Ensure `NCCL` environment variables are optimized (`NCCL_P2P_DISABLE`, `NCCL_SOCKET_IFNAME`, etc.)
- Use `torch.cuda.set_device(local_rank)` inside each process.
- Always set `local_rank`, `world_size`, `master_port` from CLI or via script args.

---

## Output Locations
Each script saves results to a subdirectory under `output/`, including checkpoints and sample images.