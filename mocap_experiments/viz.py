'''
Taken from: https://github.com/magnux/MotionGAN/blob/master/utils/viz.py
'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from PIL import Image


H36_BODY_MEMBERS_FULL = {
    'left_arm': {'joints': [13, 16, 17, 18, 19, 20, 21], 'side': 'left'},
    'left_fingers': {'joints': [19, 22, 23], 'side': 'left'},
    'right_arm': {'joints': [13, 24, 25, 26, 27, 28, 29], 'side': 'right'},
    'right_fingers': {'joints': [27, 30, 31], 'side': 'right'},
    'head': {'joints': [13, 14, 15], 'side': 'right'},
    'torso': {'joints': [0, 11, 12, 13], 'side': 'right'},
    'left_leg': {'joints': [0, 6, 7, 8, 9, 10], 'side': 'left'},
    'right_leg': {'joints': [0, 1, 2, 3, 4, 5], 'side': 'right'},
}
H36_NJOINTS_FULL = 32

H36M_USED_JOINTS = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

H36_BODY_MEMBERS = {
    'left_arm': {'joints': [13, 17, 18, 19], 'side': 'left'},
    'right_arm': {'joints': [13, 25, 26, 27], 'side': 'right'},
    'head': {'joints': [13, 14, 15], 'side': 'right'},
    'torso': {'joints': [0, 12, 13], 'side': 'right'},
    'left_leg': {'joints': [0, 6, 7, 8], 'side': 'left'},
    'right_leg': {'joints': [0, 1, 2, 3], 'side': 'right'},
}

H36_ACTIONS = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',
               'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking',
               'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']

OPENPOSE_BODY_MEMBERS = {
    'left_arm': {'joints': [2, 3, 4, 3, 2], 'side': 'left'},
    'right_arm': {'joints': [5, 6, 7, 6, 5], 'side': 'right'},
    'head': {'joints': [1, 0, 1], 'side': 'right'},
    # 'ext_head': {'joints': [14, 15, 16, 17, 16, 15, 14], 'side': 'right'},
    'ears': {'joints': [14, 0, 15], 'side': 'right'},
    'torso': {'joints': [2, 1, 5, 1, 8, 1, 11], 'side': 'right'},
    'left_leg': {'joints': [8, 9, 10, 9, 8], 'side': 'left'},
    'right_leg': {'joints': [11, 12, 13, 12, 11], 'side': 'right'},
}
OPENPOSE_NJOINTS = 16


def select_dataset(data_set):

    if data_set == "NTURGBD":
        raise NotImplementedError()
    elif data_set == "MSRC12":
        raise NotImplementedError()
    elif data_set == "Human36":
        actions_l = H36_ACTIONS
        njoints = len(H36M_USED_JOINTS)
        body_members = H36_BODY_MEMBERS
        new_body_members = {}
        for key, value in body_members.items():
            new_body_members[key] = value.copy()
            new_body_members[key]['joints'] = [H36M_USED_JOINTS.index(j) for j in new_body_members[key]['joints']]
        body_members = new_body_members
    else:
        raise NotImplementedError()

    return actions_l, njoints, body_members


class Ax3DPose(object):
    def __init__(self, ax, data_set, lcolor="#3498db", rcolor="#e74c3c"):
        """
        Create a 3d pose visualizer that can be updated with new poses.
        Args
          ax: 3d axis to plot the 3d pose on
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """

        _, self.njoints, self.body_members = select_dataset(data_set)

        self.ax = ax

        # Make connection matrix
        self.plots = {}
        for member in self.body_members.values():
            for j in range(len(member['joints']) - 1):
                j_idx_start = member['joints'][j]
                j_idx_end = member['joints'][j + 1]
                self.plots[(j_idx_start, j_idx_end)] = \
                    self.ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=lcolor if member['side'] == 'left' else rcolor)

        self.plots_mask = []
        for j in range(self.njoints):
            self.plots_mask.append(
                self.ax.plot([0], [0], [0], lw=2, c='black', markersize=8, marker='o',
                             linestyle='dashed', visible=False))

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.axes_set = False

    def update(self, channels, r_base=1000):
        """
        Update the plotted 3d pose.
        Args:
          channels: njoints * 3-dim long np array. The pose to plot.
          lcolor: String. Colour for the left part of the body.
          rcolor: String. Colour for the right part of the body.
        Returns:
          Nothing. Simply updates the axis with the new pose.
        """

        assert channels.size == self.njoints * 3, \
            "channels should have %d entries, it has %d instead" % (self.njoints * 3, channels.size)
        vals = np.reshape(channels, (self.njoints, -1))

        for member in self.body_members.values():
            for j in range(len(member['joints']) - 1):
                j_idx_start = member['joints'][j]
                j_idx_end = member['joints'][j + 1]
                x = np.array([vals[j_idx_start, 0], vals[j_idx_end, 0]])
                y = np.array([vals[j_idx_start, 1], vals[j_idx_end, 1]])
                z = np.array([vals[j_idx_start, 2], vals[j_idx_end, 2]])
                self.plots[(j_idx_start, j_idx_end)][0].set_xdata(x)
                self.plots[(j_idx_start, j_idx_end)][0].set_ydata(y)
                self.plots[(j_idx_start, j_idx_end)][0].set_3d_properties(z)

        if not self.axes_set:
            r = r_base
            xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
            # xroot, yroot, zroot = 0, 0, vals[0, 2]
            self.ax.set_xlim3d([-r + xroot, r + xroot])
            self.ax.set_zlim3d([-r + zroot, r + zroot])
            self.ax.set_ylim3d([-r + yroot, r + yroot])

            self.ax.set_aspect('equal')
            self.axes_set = True


