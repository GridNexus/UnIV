# 创建 4 个 tmux 窗口
tmux new-session -d -s train0 -n "loss_ada"
tmux new-session -d -s train1 -n "loss_l1"  
tmux new-session -d -s train2 -n "loss_stat"
tmux new-session -d -s train3 -n "structure_v4"

# 在每个窗口运行对应命令
tmux send-keys -t train0 "CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 train_conv_adaweight.py" C-m

tmux send-keys -t train1 "CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29501 train_conv_l1loss.py" C-m

tmux send-keys -t train2 "CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29502 train_conv_statloss.py" C-m

tmux send-keys -t train3 "CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29503 train_structure_ep100_layernorm_v4.py" C-m

# 查看所有会话
tmux ls

#  attach 到具体会话查看输出
# tmux attach -t train0
# 按 Ctrl+B 然后 D  detach

# 杀死所有会话
# tmux kill-server