import io
import tensorflow as tf
import matplotlib.pyplot as plt


class TensorboardLogger(object):
    '''
    Functions which can be used to log numpy objects 
    during training.
    '''

    def __init__(self, path, graph, no_log=False):
        if no_log is True:
            print('warning not logging this run!!')
        else:
            self.summary_writer = tf.summary.FileWriter(logdir=path, graph=graph)
        self.no_log = no_log

    def add_tf_summary(self, summary, np_step):
        if not self.no_log:
            self.summary_writer.add_summary(summary, global_step=np_step)

    def add_np_scalar(self, np_scalar, np_step, tag):
        '''
        Add a numpy scalar variable to the tensorboard summary.
        Args:
            np_scalar: A ()-shape numpy array
            no_step: A ()-shaped numpy array
            tag: A string under which the data will show up in tensorboard.
        '''
        if not self.no_log:
            summary = tf.Summary.Value(tag=tag, simple_value=np_scalar)
            summary = tf.Summary(value=[summary])
            self.summary_writer.add_summary(summary, global_step=np_step)

    def add_np_plot(self, np_array, np_step, tag):
        '''
        Add a numpy array as plot to tensorboard
        Args:
            np_array: A (n)-shape numpy array
            no_step: A ()-shaped numpy array
            tag: A string under which the data will show up in tensorboard.
        '''
        # window plot in tensorboard.
        if not self.no_log:
            plt.figure()
            plt.plot(np_array)
            plt.title(tag)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            summary_image = tf.Summary.Image(
                encoded_image_string=buf.getvalue(),
                height=int(plt.rcParams["figure.figsize"][0]*100),
                width=int(plt.rcParams["figure.figsize"][1]*100))
            summary_image = tf.Summary.Value(tag=tag,
                                             image=summary_image)
            summary_image = tf.Summary(value=[summary_image])
            self.summary_writer.add_summary(summary_image, global_step=np_step)
            plt.close()
            buf.close()

    def add_np_image(self, np_matrix, np_step, tag):
        '''
        Add a numpy array as plot to tensorboard
        Args:
            np_matrix: A (n, m)-shape numpy array
            no_step: A ()-shaped numpy array
            tag: A string under which the data will show up in tensorboard.
        '''
        # window plot in tensorboard.
        if not self.no_log:
            plt.figure()
            plt.imshow(np_matrix)
            plt.title(tag)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            summary_image = tf.Summary.Image(
                encoded_image_string=buf.getvalue(),
                height=int(plt.rcParams["figure.figsize"][0]*100),
                width=int(plt.rcParams["figure.figsize"][1]*100))
            summary_image = tf.Summary.Value(tag=tag,
                                             image=summary_image)
            summary_image = tf.Summary(value=[summary_image])
            self.summary_writer.add_summary(summary_image, global_step=np_step)
            plt.close()
            buf.close()
