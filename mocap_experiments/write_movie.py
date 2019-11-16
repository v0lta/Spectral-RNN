'''
Take a look at
https://github.com/magnux/MotionGAN/blob/master/utils/human36_skels_to_h5.py
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import mocap_experiments.viz as viz
import cv2
from mpl_toolkits.mplot3d import Axes3D


def read_pose(x):
    pose = np.reshape(np.transpose(np.array(x), [2, 0, 1]), [32, 3, -1])
    pose_x = pose[:, 0, :]
    pose_y = pose[:, 1, :]
    pose_z = pose[:, 2, :]
    return np.stack([pose_x, pose_y, pose_z], axis=1)


def write_movie(pose_array, name='writer_test.mp4', fps=15, r_base=1000, color_shift_at=0):
    """
    :param pose_array: np.array [joints, 3dpoints, time]
    :param name: the name for the movie.
    """
    time = pose_array.shape[-1]
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig = plt.figure()
    with writer.saving(fig, name, 100):
        ax = fig.add_subplot(111, projection='3d')
        gt_plotter = viz.Ax3DPose(ax, 'Human36')
        net_plotter = viz.Ax3DPose(ax, 'Human36', lcolor='#e88d1e', rcolor='#3ce7ae')
        net_plotter.axes_set = True
        for i in range(time):
            if i > color_shift_at:
                net_plotter.update(pose_array[:, :, i].flatten(), r_base=r_base)
            else:
                gt_plotter.update(pose_array[:, :, i].flatten(), r_base=r_base)
            writer.grab_frame()
    plt.close()


def write_figure(pose_array, name='writer_test.pdf', r_base=1000, color_shift_at=0):
    time = pose_array.shape[-1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(time):
        if i > color_shift_at:
            plotter = viz.Ax3DPose(ax, 'Human36', lcolor='#e88d1e', rcolor='#3ce7ae')
            plotter.update(pose_array[:, :, i].flatten(), r_base=r_base)
        else:
            plotter = viz.Ax3DPose(ax, 'Human36')
            plotter.update(pose_array[:, :, i].flatten(), r_base=r_base)
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xticklabels([])
    plt.savefig(name)
    plt.close()

def write_subplots(pose_array, name='writer_test.pdf', r_base=1000, color_shift_at=0):
    time = pose_array.shape[-1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(time):
        if i > color_shift_at:
            plotter = viz.Ax3DPose(ax, 'Human36', lcolor='#e88d1e', rcolor='#3ce7ae')
            plotter.update(pose_array[:, :, i].flatten(), r_base=r_base)
        else:
            plotter = viz.Ax3DPose(ax, 'Human36')
            plotter.update(pose_array[:, :, i].flatten(), r_base=r_base)
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xticklabels([])
    plt.savefig(name)
    plt.close()

if __name__ == "__main__":
    os.environ["CDF_LIB"] = '/home/moritz/CDF/cdf37_0-dist/lib'
    from spacepy import pycdf

    # test_file = '/home/moritz/uni/freq_loss_H3.6M/data/' \
    #             'un_zipped/by_action/Poses_D3_Positions_mono_universal_Directions/' \
    #             'S1/MyPoseFeatures/D3_Positions_mono_universal/Directions 1.54138969.cdf'
    test_file = '/home/moritz/uni/freq_loss_H3.6M/data/' \
                'un_zipped/by_actor/S1/MyPoseFeatures/D3_Positions/Walking.cdf'

    with pycdf.CDF(test_file) as cdffile:
        pose = read_pose(cdffile['Pose'].copy())
        pose = pose[viz.H36M_USED_JOINTS, :, :]
        print(pose.shape)

    total_time = 400
    color_shift_at = 200
    write_movie(pose[:, :, :400], color_shift_at=200)
    write_figure(pose[:, :, :400:10], color_shift_at=int(200/10))
