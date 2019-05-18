#block(name=nips_power_fft, threads=2, memory=7500, subtasks=1, hours=48, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython power_train_exp.py
   
