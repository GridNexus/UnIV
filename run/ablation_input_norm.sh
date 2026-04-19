torchrun --nproc_per_node=2 train_ddp_rmsnorm.py
torchrun --nproc_per_node=2 train_ddp_layernorm.py
torchrun --nproc_per_node=2 train_ddp_none.py
# torchrun --nproc_per_node=1 test.py