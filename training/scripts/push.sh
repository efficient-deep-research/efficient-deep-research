singularity exec \
    --network host \
    --writable-tmpfs \
    --bind $PBS_O_WORKDIR/download:/mnt/workspace \
    "$PBS_O_WORKDIR/container/ms-swift_container.sif" \
    swift export \
        --adapters $PBS_O_WORKDIR/output/1276612.pbs1/v0-20251021-223456/checkpoint-20 \
        --push_to_hub true \
        --use_hf true \
        --hub_model_id 'efficient-deep-research/gap_0_5_lora_ckpt_20' \
        --hub_token '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'