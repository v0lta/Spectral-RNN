"""
Partially taken from https://github.com/magnux/MotionGAN/blob/master/utils/npangles.py
and https://github.com/magnux/MotionGAN/blob/master/utils/seq_utils.py
"""
import numpy as np
from collections import OrderedDict
from mocap_experiments.viz import H36_BODY_MEMBERS


def organize_into_batches(batches, pd):
    batch_total = len(batches)
    split_into = int(batch_total/pd['batch_size'])
    stop_at = pd['batch_size']*split_into
    batch_lst = np.array_split(np.stack(batches[:stop_at]),
                               split_into)
    return batch_lst


def pd_to_string(pd_var) -> str:
    '''
    Convert a parameter dict to string
    :param pd_var: The Parameter dictionary
    :return: A string containg what was in the dict.
    '''
    pd_var = pd_var.copy()
    pd_var_str = ''
    for key in list(pd_var.keys()):
        if type(pd_var[key]) is str:
            pd_var_str += '_' + key + pd_var[key]
        elif type(pd_var[key]) is bool:
            if pd_var[key] is True:
                pd_var_str += '_' + key
        elif type(pd_var[key]) is list:
            pd_var_str += '_' + key + str(pd_var[key][0])
        else:
            pd_var_str += '_' + key + str(pd_var[key])
    return pd_var_str


def get_body_graph(body_members):
    new_body_members = {}
    used_joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
    for key, value in body_members.items():
        new_body_members[key] = value.copy()
        new_body_members[key]['joints'] = [used_joints.index(j) for j in new_body_members[key]['joints']]
    body_members = new_body_members

    body_members = OrderedDict(sorted(body_members.items()))

    members_from = []
    members_to = []
    for member in body_members.values():
        for j in range(len(member['joints']) - 1):
            members_from.append(member['joints'][j])
            members_to.append(member['joints'][j + 1])

    members_lst = list(zip(members_from, members_to))

    graph = {name: set() for tup in members_lst for name in tup}
    has_parent = {name: False for tup in members_lst for name in tup}
    for parent, child in members_lst:
        graph[parent].add(child)
        has_parent[child] = True

    # roots = [name for name, parents in has_parent.items() if not parents]  # assuming 0 (hip)

    def traverse(hierarchy, graph, names):
        for name in names:
            hierarchy[name] = traverse({}, graph, graph[name])
        return hierarchy

    # print(traverse({}, graph, roots))

    for key, value in graph.items():
        graph[key] = sorted(list(graph[key]))

    return members_from, members_to, graph


def quaternion_between(u, v):
    """
    Finds the quaternion between two tensor of 3D vectors.
    See:
    http://www.euclideanspace.com/maths/algebra/vectors/angleBetween/index.htm
    Args:
        u: A `np.array` of rank R, the last dimension must be 3.
        v: A `np.array` of rank R, the last dimension must be 3.
    Returns:
        A `Quaternion` of Rank R with the last dimension being 4.
        returns 1, 0, 0, 0 quaternion if either u or v is 0, 0, 0
    Raises:
        ValueError, if the last dimension of u and v is not 3.
    """
    if u.shape[-1] != 3 or v.shape[-1] != 3:
        raise ValueError("The last dimension of u and v must be 3.")

    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")

    def _vector_batch_dot(a, b):
        return np.sum(np.multiply(a, b), axis=-1, keepdims=True)

    def _length_2(a):
        return np.sum(np.square(a), axis=-1, keepdims=True)

    def _normalize(a):
        return a / np.sqrt(_length_2(a) + 1e-8)

    base_shape = [int(d) for d in u.shape]
    base_shape[-1] = 1
    zero_dim = np.zeros(base_shape)
    one_dim = np.ones(base_shape)
    w = np.sqrt(_length_2(u) * _length_2(v)) + _vector_batch_dot(u, v)

    q = np.where(
        np.tile(np.equal(np.sum(u, axis=-1, keepdims=True), zero_dim), [1 for _ in u.shape[:-1]] + [4]),
        np.concatenate([one_dim, u], axis=-1),
        np.where(
            np.tile(np.equal(np.sum(v, axis=-1, keepdims=True), zero_dim), [1 for _ in u.shape[:-1]] + [4]),
            np.concatenate([one_dim, v], axis=-1),
            np.where(
                np.tile(np.less(w, 1e-4), [1 for _ in u.shape[:-1]] + [4]),
                np.concatenate([zero_dim, np.stack([-u[..., 2], u[..., 1], u[..., 0]], axis=-1)], axis=-1),
                np.concatenate([w, np.cross(u, v)], axis=-1)
            )
        )
    )

    return _normalize(q)


def expmap_to_rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
      r: (..., 3) exponential map Tensor
    Returns:
      R: (..., 3, 3) rotation matrix Tensor
    """
    base_shape = [int(d) for d in r.shape][:-1]
    zero_dim = np.zeros(base_shape)

    theta = np.sqrt(np.sum(np.square(r), axis=-1, keepdims=True) + 1e-8)
    r0 = r / theta

    r0x = np.reshape(
        np.stack([zero_dim, -1.0 * r0[..., 2], r0[..., 1],
                  zero_dim, zero_dim, -1.0 * r0[..., 0],
                  zero_dim, zero_dim, zero_dim], axis=-1),
        base_shape + [3, 3]
    )
    trans_dims = list(range(len(r0x.shape)))
    trans_dims[-1], trans_dims[-2] = trans_dims[-2], trans_dims[-1]
    r0x = r0x - np.transpose(r0x, trans_dims)

    tile_eye = np.tile(np.reshape(np.eye(3), [1 for _ in base_shape] + [3, 3]), base_shape + [1, 1])
    theta = np.expand_dims(theta, axis=-1)

    R = tile_eye + np.sin(theta) * r0x + (1.0 - np.cos(theta)) * np.matmul(r0x, r0x)
    return R


def quaternion_to_expmap(q):
    """
    Converts a quaternion to an exponential map
    Tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
        q: (..., 4) quaternion Tensor
    Returns:
        r: (..., 3) exponential map Tensor
    Raises:
        ValueError if the l2 norm of the quaternion is not close to 1
    """
    # if (np.abs(np.linalg.norm(q)-1)>1e-3):
    # raise(ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta = np.sqrt(np.sum(np.square(q[..., 1:]), axis=-1, keepdims=True) + 1e-8)
    coshalftheta = np.expand_dims(q[..., 0], axis=-1)

    r0 = q[..., 1:] / sinhalftheta
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)

    condition = np.greater(theta, np.pi)
    theta = np.where(condition, 2 * np.pi - theta, theta)
    r0 = np.where(np.tile(condition, [1 for _ in condition.shape[:-1]] + [3]), -r0, r0)
    r = r0 * theta

    return r


def rotmat_to_euler(R):
    """
    Converts a rotation matrix to Euler angles
    Tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
      R: a (..., 3, 3) rotation matrix Tensor
    Returns:
      eul: a (..., 3) Euler angle representation of R
    """
    base_shape = [int(d) for d in R.shape][:-2]
    zero_dim = np.zeros(base_shape)
    one_dim = np.ones(base_shape)

    econd0 = np.equal(R[..., 0, 2], one_dim)
    econd1 = np.equal(R[..., 0, 2], -1.0 * one_dim)
    econd = np.logical_or(econd0, econd1)

    e2 = np.where(
        econd,
        np.where(econd1, one_dim * np.pi / 2.0, one_dim * -np.pi / 2.0),
        -np.arcsin(R[..., 0, 2])
    )
    e1 = np.where(
        econd,
        np.arctan2(R[..., 1, 2], R[..., 0, 2]),
        np.arctan2(R[..., 1, 2] / np.cos(e2), R[..., 2, 2] / np.cos(e2))
    )
    e3 = np.where(
        econd,
        zero_dim,
        np.arctan2(R[..., 0, 1] / np.cos(e2), R[..., 0, 0] / np.cos(e2))
    )

    eul = np.stack([e1, e2, e3], axis=-1)
    return eul


def seq_to_angles_transformer(body_members):
    """
    As found at: https://github.com/magnux/MotionGAN/blob/master/utils/seq_utils.py
    """
    _, _, body_graph = get_body_graph(body_members)

    def _get_angles(coords):
        base_shape = [int(dim) for dim in coords.shape]
        base_shape.pop(1)
        base_shape[-1] = 1

        coords_list = np.split(coords, int(coords.shape[1]), axis=1)
        coords_list = [np.squeeze(elem, axis=1) for elem in coords_list]

        def _get_angle_for_joint(joint_idx, parent_idx, angles):
            if parent_idx is None:  # joint_idx should be 0
                parent_bone = np.concatenate([np.ones(base_shape),
                                              np.zeros(base_shape),
                                              np.zeros(base_shape)], axis=-1)
            else:
                parent_bone = coords_list[parent_idx] - coords_list[joint_idx]

            for child_idx in body_graph[joint_idx]:
                child_bone = coords_list[child_idx] - coords_list[joint_idx]
                angle = quaternion_between(parent_bone, child_bone)
                angle = quaternion_to_expmap(angle)
                angle = expmap_to_rotmat(angle)
                angle = rotmat_to_euler(angle)
                angles.append(angle)

            for child_idx in body_graph[joint_idx]:
                angles = _get_angle_for_joint(child_idx, joint_idx, angles)

            return angles

        angles = _get_angle_for_joint(0, None, [])
        fixed_angles = len(body_graph[0])
        angles = angles[fixed_angles:]
        return np.stack(angles, axis=1)

    return _get_angles


def compute_ent_metrics(gt_seqs, seqs, print_debug=False):
    """
    :param numpy.array gt_seqs: The ground truth sequences, [batch_size, njoints, seq_len, 3]
    :param numpy.array seqs: The generated sequence  [batch_size, njoints, seq_len, 3]
    :param int seq_len: The length of both sequences [seq_len]
    :return Spectral entropy and kl divergence of both combinations.
    """
    angle_trans = seq_to_angles_transformer(H36_BODY_MEMBERS)
    gt_cent_seqs = gt_seqs - gt_seqs[:, 0, np.newaxis, :, :]
    gt_angle_expmaps = angle_trans(gt_cent_seqs)
    cent_seqs = seqs - seqs[:, 0, np.newaxis, :, :]
    angle_expmaps = angle_trans(cent_seqs)

    gt_angle_seqs = rotmat_to_euler(expmap_to_rotmat(gt_angle_expmaps))
    angle_seqs = rotmat_to_euler(expmap_to_rotmat(angle_expmaps))

    gt_seqs_fft = np.fft.fft(gt_angle_seqs, axis=2)
    gt_seqs_ps = np.abs(gt_seqs_fft) ** 2

    gt_seqs_ps_global = gt_seqs_ps.sum(axis=0) + 1e-8
    gt_seqs_ps_global /= gt_seqs_ps_global.sum(axis=1, keepdims=True)

    seqs_fft = np.fft.fft(angle_seqs, axis=2)
    seqs_ps = np.abs(seqs_fft) ** 2

    seqs_ps_global = seqs_ps.sum(axis=0) + 1e-8
    seqs_ps_global /= seqs_ps_global.sum(axis=1, keepdims=True)

    seqs_ent_global = -np.sum(seqs_ps_global * np.log(seqs_ps_global), axis=1)
    if print_debug:
        print("PS Entropy: ", seqs_ent_global.mean())

    seqs_kl_gen_gt = np.sum(seqs_ps_global * np.log(seqs_ps_global / gt_seqs_ps_global), axis=1)
    if print_debug:
        print("PS KL(Gen|GT): ", seqs_kl_gen_gt.mean())
    seqs_kl_gt_gen = np.sum(gt_seqs_ps_global * np.log(gt_seqs_ps_global / seqs_ps_global), axis=1)
    if print_debug:
            print("PS KL(GT|Gen): ", seqs_kl_gt_gen.mean())
    return seqs_ent_global.mean(), seqs_kl_gen_gt.mean(), seqs_kl_gt_gen.mean()


def compute_ent_metrics_splits(gt_seqs, seqs, seq_len, print_debug=True):
    """
    As found at  https://github.com/magnux/MotionGAN/blob/master/test.py line 634.
    """
    seqs_ent_global_lst = []
    seqs_kl_gen_get_lst = []
    seqs_kl_gt_gen_lst = []
    for seq_start, seq_end in [(s * (seq_len // 4), (s + 1) * (seq_len // 4)) for s in range(4)] + [(0, seq_len)]:
        gt_seqs_tmp = gt_seqs[:, :, seq_start:seq_end, :]
        seqs_tmp = seqs[:, :, seq_start:seq_end, :]

        seqs_ent_global_mean, seqs_kl_gen_gt_mean, seqs_kl_gt_gen_mean = \
            compute_ent_metrics(gt_seqs=gt_seqs_tmp, seqs=seqs_tmp, print_debug=print_debug)

        print("frames: ", (seq_start, seq_end),
              "%.5f & %.5f & %.5f" % (seqs_ent_global_mean, seqs_kl_gen_gt_mean, seqs_kl_gt_gen_mean))
        seqs_ent_global_lst.append(seqs_ent_global_mean)
        seqs_kl_gen_get_lst.append(seqs_kl_gen_gt_mean)
        seqs_kl_gt_gen_lst.append(seqs_kl_gt_gen_mean)
    return seqs_ent_global_lst, seqs_kl_gen_get_lst, seqs_kl_gt_gen_lst


if __name__ == '__main__':
    import collections
    from mocap_experiments.load_h36m import H36MDataSet

    for seq_len in [100]:
        PoseData = collections.namedtuple('PoseData', ['f', 'action', 'actor', 'array'])
        data_set = H36MDataSet(chunk_size=seq_len, dataset_name='h36m')
        data_set_val = H36MDataSet(chunk_size=seq_len, dataset_name='h36m', train=False)
        train_batches = data_set.data_array
        val_batches = data_set_val.data_array
        # compute metric on two.
        # self.batch_size, self.njoints, self.seq_len, 3
        _ = compute_ent_metrics_splits(gt_seqs=np.moveaxis(train_batches, [0, 1, 2, 3], [0, 2, 1, 3]),
                                       seqs=np.moveaxis(val_batches, [0, 1, 2, 3], [0, 2, 1, 3]),
                                       seq_len=seq_len, print_debug=True)

    # 50  0.6601937642721489 0.0054249518712179545 0.004659657744025847
    # 100 1.008094971949801 0.006738225054610497 0.006051738150345249
    # 150 1.2329514686699823 0.007361603968491055 0.006864819543108966
    # 200 1.4182707403871782 0.008593273992032848 0.00806628118754446
    # pass