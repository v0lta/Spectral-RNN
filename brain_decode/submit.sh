#block(name=nips_brain, threads=2, memory=7500, subtasks=1, hours=96, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython train_class_cnn.py
   
