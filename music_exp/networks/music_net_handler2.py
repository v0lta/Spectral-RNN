import numpy as np
from subprocess import call
import os
import os.path
import errno
import csv
from six.moves import urllib
import random
from intervaltree import IntervalTree
from scipy.io import wavfile
from IPython.core.debugger import Tracer
debug_here = Tracer()

sz_float = 4     # size of a float
epsilon = 10e-8  # fudge factor for normalization


class MusicNet(object):
    """`MusicNet <http://homes.cs.washington.edu/~thickstn/musicnet.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset
        train (bool, optional): If True, creates dataset from ``train_data``,
            otherwise from ``test_data``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        mmap (bool, optional): If true, mmap the dataset for faster access times.
        normalize (bool, optional): If true, rescale input vectors to unit norm.
        window (int, optional): Size in samples of a data point.
        pitch_shift (int,optional): Integral pitch-shifting transformations.
        jitter (int, optional): Continuous pitch-jitter transformations.
        epoch_size (int, optional): Designated Number of samples for an "epoch"
    """
    url = 'https://homes.cs.washington.edu/~thickstn/media/musicnet.tar.gz'
    raw_folder = 'raw'
    train_data, train_labels, train_tree = 'train_data', 'train_labels', 'train_tree.pckl'
    test_data, test_labels, test_tree = 'test_data', 'test_labels', 'test_tree.pckl'
    extracted_folders = [train_data, train_labels, test_data, test_labels]

    def __init__(self, root, train=True, download=False, normalize=True,
                 window=16384, pitch_shift=0, jitter=0., random_training=False):
        self.normalize = normalize
        self.window = window
        self.pitch_shift = pitch_shift
        self.jitter = jitter
        self.m = 128

        self.root = os.path.expanduser(root)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        # open the raw data.
        self.test_records, self.test_labels, self.train_records, self.train_labels = \
            self.open_raw_data_set()

        self.train_keys = self.train_records.keys()
        self.test_keys = self.test_records.keys()

    def get_train_batches(self, batch_size):
        train_xy, window_total, sample_total = self.get_train_data()
        assert len(train_xy) == window_total
        x_batches = []
        y_batches = []
        batch_count = 0
        x_batch = []
        y_batch = []
        for x, y in train_xy:
            x_batch.append(x)
            y_batch.append(y)
            batch_count += 1
            if batch_count == batch_size:
                x_batches.append(np.stack(x_batch, axis=0))
                y_batches.append(np.stack(y_batch, axis=0))
                batch_count = 0
                x_batch = []
                y_batch = []
        return x_batches, y_batches

    def get_train_data(self):
        return self.get_data(self.train_records, self.train_labels, randomize=True)

    def get_data(self, records, labels, randomize=True):
        '''
        Get the training samples, estimate mean and variance.
        '''
        # estimate data_set size mean and variance.
        window_total = 0
        sample_total = 0
        # train_mean = 0  # approx zero :-D.
        train_xy = []
        for record_key in self.train_keys:
            # get the number of windows for this point, -2 for wiggle room.
            current_record = self.train_records[record_key]
            assert len(current_record.shape) == 1
            rec_length = current_record.shape[0]
            # rec_mean = np.mean(current_record)
            rec_windows = int(rec_length/self.window) - 2
            window_total += rec_windows
            sample_total += rec_length
            # train_mean += rec_mean*rec_length
            # move around randomly in the first window, to change
            # the samples slightly.
            if randomize:
                offset = np.random.randint(0, self.window)
            else:
                offset = self.window
            for i in range(rec_windows):
                start = offset+i*self.window
                x = current_record[start:(start+self.window)]
                y = np.zeros(self.m, dtype=np.float32)
                for label in self.train_labels[record_key][start+int(self.window/2)]:
                    y[label.data[1]] = 1
                train_xy.append((x, y))

        # train_mean = train_mean/sample_total
        self.size = window_total
        self.sample_total = sample_total  # The total number of samples in train.
        if randomize:
            random.shuffle(train_xy)  # shuffle works in place and returns none.
        return train_xy, window_total, sample_total

    def get_test_data(self):
        return self.get_data(self.test_records, self.test_labels, randomize=False)

    #  TODO include shifts and so on...
    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): (ignored by this dataset; a random data point is returned)
    #     Returns:
    #         tuple: (audio, target) where target is a binary
    #                 vector indicating notes on at the center of the audio.
    #     """

    #     shift = 0
    #     if self.pitch_shift > 0:
    #         shift = np.random.randint(-self.pitch_shift, self.pitch_shift)

    #     jitter = 0.
    #     if self.jitter > 0:
    #         jitter = np.random.uniform(-self.jitter, self.jitter)

    #     rec_id = self.rec_ids[np.random.randint(0, len(self.rec_ids))]
    #     s = np.random.randint(0, self.records[rec_id][1]
    #                           - (2.**((shift+jitter)/12.)) * self.window)
    #     return self.access(rec_id, s, shift, jitter)

    def __len__(self):
        return self.size

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.train_data)) and \
            os.path.exists(os.path.join(self.root, self.test_data)) and \
            os.path.exists(os.path.join(self.root, self.train_labels,
                                        self.train_tree)) and \
            os.path.exists(os.path.join(self.root, self.test_labels, self.test_tree))

    def download(self):
        """Download the MusicNet data if it doesn't exist in ``raw_folder`` already."""

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        filename = self.url.rpartition('/')[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        debug_here()
        if not os.path.exists(file_path):
            print('Downloading ' + self.url)
            data = urllib.request.urlopen(self.url)
            with open(file_path, 'wb') as f:
                f.write(data.read())
        if not all(map(lambda f: os.path.exists(os.path.join(self.root, f)),
                       self.extracted_folders)):
            print('Extracting ' + filename)
            if call(["tar", "-xvvf", file_path]) != 0:
                raise OSError("Failed tarball extraction")

        print('Download Complete')

    def open_raw_data_set(self):
        "load the raw data into memory."
        test_records = self.load_wav(self.test_data)
        test_labels = self.load_labels(self.test_labels)
        train_records = self.load_wav(self.train_data)
        train_labels = self.load_labels(self.train_labels)
        return test_records, test_labels, train_records, train_labels

    def load_wav(self, path):
        # read wav-files into memory.
        return_dict = {}
        for item in os.listdir(os.path.join(self.root, path)):
            if not item.endswith('.wav'):
                continue
            uid = int(item[:-4])
            _, data = wavfile.read(os.path.join(self.root, path, item))
            return_dict[int(uid)] = data
        return return_dict

    # wite out labels in intervaltrees for fast access
    def load_labels(self, path):
        trees = dict()
        for item in os.listdir(os.path.join(self.root, path)):
            if not item.endswith('.csv'):
                continue
            uid = int(item[:-4])
            tree = IntervalTree()
            with open(os.path.join(self.root, path, item), 'rt') as f:
                reader = csv.DictReader(f, delimiter=',')
                for label in reader:
                    start_time = int(label['start_time'])
                    end_time = int(label['end_time'])
                    instrument = int(label['instrument'])
                    note = int(label['note'])
                    start_beat = float(label['start_beat'])
                    end_beat = float(label['end_beat'])
                    note_value = label['note_value']
                    tree[start_time:end_time] = (instrument, note, start_beat,
                                                 end_beat, note_value)
            trees[uid] = tree
        return trees
