#block(name=tmux-interactive, threads=2, memory=7500, subtasks=1, hours=48, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    tmux new -s terminal -d
    sleep 24h
   
