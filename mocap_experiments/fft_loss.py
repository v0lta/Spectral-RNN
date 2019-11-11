import tensorflow as tf
import collections
PoseData = collections.namedtuple('PoseData', ['f', 'action', 'actor', 'array'])


def compute_power_spectrum(x):
    """
    Compute the power spectrum |F(x)| of x.
    :param x: Mocap-data tensor [batch_size, time, 17*3]
    :return: The squared radii of the fourier coefficients as in r of r e^(i phi) [batch_size, 17*3, radii]
    """
    x = tf.transpose(x, [0, 2, 1])
    x = tf.complex(x, tf.zeros_like(x))
    freq_x = tf.signal.fft(x)
    ps_x = tf.real(freq_x) * tf.real(freq_x) + tf.imag(freq_x) * tf.imag(freq_x)
    return ps_x


def power_spectrum_entropy(x):
    """
    :param x: Mocap-data tensor [batch_size, time, 17*3]
    :return: A scalar containing the power spectrum entropy [batch_size, 17*3=51]
    """
    ps = compute_power_spectrum(x) + 1e-8  # add a small epsilon for numerical stability.
    pm = ps/tf.expand_dims(tf.math.reduce_sum(ps, axis=-1), -1)
    entropy_h = (-1)*tf.math.reduce_sum(pm*tf.log(pm), axis=-1)
    return entropy_h


def power_spectrum_kl_divergence(x, y):
    """
    KL divergence of the power spectra of x and y.
    :param x: Mocap-data tensor x [batch_size, time, 17*3]
    :param y: Mocap-data tensor y [batch_size, time, 17*3]
    :return: ps_kl_xy, ps_kl_yx both [batch_size, 17*3]
    """
    psx = compute_power_spectrum(x) + 1e-8
    psy = compute_power_spectrum(y) + 1e-8
    # psx_dist = tf.nn.softmax(psx) + 1e-8
    # psy_dist = tf.nn.softmax(psy) + 1e-8
    psx_dist = psx/tf.expand_dims(tf.reduce_sum(psx, axis=-1), -1)
    psy_dist = psy/tf.expand_dims(tf.reduce_sum(psy, axis=-1), -1)
    ps_kl_xy = tf.reduce_sum(psx_dist*tf.log(psx_dist/psy_dist), axis=-1)
    ps_kl_yx = tf.reduce_sum(psy_dist*tf.log(psy_dist/psx_dist), axis=-1)
    return ps_kl_xy, ps_kl_yx


def consistency_loss_fun(x, y, lambda_a=0, lambda_b=1,  summary_nodes=False):
    """
    Set up a differentiable consistency loss.
    :param x: Mocap-data tensor x [batch_size, time, 17*3]
    :param y: Mocap-data tensor y [batch_size, time, 17*3]
    :return: The power spectrum entropy and kl-divergence based frequency domain loss.
    """
    ps_x = power_spectrum_entropy(x)
    ps_y = power_spectrum_entropy(y)
    ps_loss = ps_x - ps_y
    ps_loss = tf.reduce_mean(ps_loss*ps_loss)
    ps_kl_xy, ps_kl_yx = power_spectrum_kl_divergence(x, y)
    # pskl_loss_diff = ps_kl_xy - ps_kl_yx
    # pskl_loss = tf.reduce_mean(pskl_loss_diff*pskl_loss_diff)
    pskl_loss = tf.reduce_mean(ps_kl_yx + ps_kl_yx)
    total = ps_loss*lambda_a + pskl_loss*lambda_b
    if summary_nodes:
        tf.summary.scalar('consistencyLoss/psX', tf.reduce_mean(ps_x))
        tf.summary.scalar('consistencyLoss/psY', tf.reduce_mean(ps_y))
        tf.summary.scalar('consistencyLoss/psKlXy', tf.reduce_mean(ps_kl_xy))
        tf.summary.scalar('consistencyLoss/psKlYx', tf.reduce_mean(ps_kl_yx))
        tf.summary.scalar('consistencyLoss/psLoss', tf.reduce_mean(ps_loss))
        # tf.summary.scalar('consistencyLoss/psklLossDiff', tf.reduce_mean(pskl_loss_diff))
        tf.summary.scalar('consistencyLoss/psklLoss', tf.reduce_mean(pskl_loss))
        tf.summary.scalar('consistencyLoss/total', tf.reduce_mean(total))
    return total


if __name__ == '__main__':
    print('test mse loss')
    import numpy as np
    import matplotlib.pyplot as plt
    from mocap_experiments.write_movie import write_movie
    from mocap_experiments.load_h36m import H36MDataSet
    from mocap_experiments.util import compute_ent_metrics
    data = H36MDataSet()
    batches = (data.data_array - data.mean)/data.std
    batch_size = 30
    batch_chunk = batches[:batch_size, :, :, :]
    batch_chunk_rs = np.reshape(batch_chunk, [batch_size, data.chunk_size, 17*3])
    batch_chunk2 = batches[batch_size:(batch_size*2), :, :, :]
    batch_chunk2_rs = np.reshape(batch_chunk2, [batch_size, data.chunk_size, 17*3])
    batch_chunk_tf1 = tf.constant(batch_chunk_rs.astype(np.float32))
    batch_chunk_tf2 = tf.constant(batch_chunk2_rs.astype(np.float32))
    ps1 = power_spectrum_entropy(batch_chunk_tf1)
    pse1 = tf.reduce_mean(power_spectrum_entropy(batch_chunk_tf1))
    kl1, kl2 = power_spectrum_kl_divergence(batch_chunk_tf1, batch_chunk_tf2)
    kl1 = tf.reduce_mean(kl1)
    kl2 = tf.reduce_mean(kl2)
    loss = consistency_loss_fun(batch_chunk_tf1, batch_chunk_tf2)

    with tf.Session() as sess:
        ps1_np = sess.run(ps1)
        pse1_np = sess.run(pse1)
        kl1_np = sess.run(kl1)
        kl2_np = sess.run(kl2)
        loss_np = sess.run(loss)

    print(pse1_np, kl1_np, kl2_np, loss_np)
    print(compute_ent_metrics(np.moveaxis(batch_chunk, [0, 1, 2, 3], [0, 2, 1, 3]),
                              np.moveaxis(batch_chunk2, [0, 1, 2, 3], [0, 2, 1, 3])))
