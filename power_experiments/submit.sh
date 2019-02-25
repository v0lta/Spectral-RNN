#block(name=power_pred_fft, threads=2, memory=7500, subtasks=1, hours=24, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython power_load_pred_exp.py
   
