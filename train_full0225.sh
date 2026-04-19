# 创建 4 个 tmux 窗口
tmux new-session -d -s train_final -n "final_30epoch_var4"

# 在每个窗口运行对应命令
tmux send-keys -t train_final "torchrun --nproc_per_node=8 --master_port=29500 train_convnext_full0225.py" C-m
# 查看所有会话
tmux ls

#  attach 到具体会话查看输出
# tmux attach -t train0
# 按 Ctrl+B 然后 D  detach

# 杀死所有会话
# tmux kill-server