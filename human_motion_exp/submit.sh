#block(name=spec_loss, threads=1, memory=7500, subtasks=1, hours=8, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    # ipython src/translate.py -- --action all --seq_length_in 61 --seq_length_out 60 --GPU 0 --batch_size 16 --cgru True --fft True --window_size 30 --stiefel False --step_size 25
    # ipython src/translate.py -- --learning_rate 0.005 --omit_one_hot False --residual_velocities True --cgru False --size 1024
    ipython src/translate.py -- --learning_rate 0.005 --omit_one_hot True --cgru True --fft True --window_size 12 --step_size 6 --seq_length_in 62 --seq_length_out 25 --GPU 0 --window_fun hann --residual_velocities