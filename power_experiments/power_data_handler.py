import csv
import os
import datetime
import numpy as np
import random
import sys
sys.path.insert(0, "../")



class PowerDataHandler(object):
    '''
    Data handler meant to load and preprocess
    power load data from:
    https://transparency.entsoe.eu/
    '''

    def __init__(self, path, context=15, samples_per_day=96, test_keys=None):
        """
        Creat the power data handler
        :param path: Where to power files are stored
        :param context: The number of days used for context.
        :param samples_per_day: The number of measurement points per day.
        :param test_keys: What to test on.
        """
        self.path = path
        self.context = context
        self._samples_per_day = samples_per_day

        # walk trough subfolders of the path and find csv files.
        self.files = {}
        self.data = {}
        for root, dirs, files in os.walk(path):
            company_dict = {}
            for name in files:
                print(os.path.join(root, name))
                if name.split('.')[-1] == 'csv':
                    file_path = os.path.join(root, name)
                    with open(file_path, newline='') as csvfile:
                        power_reader = csv.reader(csvfile, delimiter=',')
                        year_lst = []
                        prev_day = '01'
                        day_data = []
                        for i, row in enumerate(power_reader):
                            current_day = row[0].split(' ')[0].split('.')[0]
                            if current_day != prev_day and i > 1:
                                year_lst.append(day_data[:self._samples_per_day])

                                day_data = []
                                # ipdb.set_trace()
                                prev_day = current_day
                                day_start = i
                            if i == 0:
                                head = row
                                pass
                            else:
                                try:
                                    forecast = int(row[1])
                                except ValueError:
                                    forecast = 0
                                try:
                                    true_value = int(row[2])
                                except ValueError:
                                    true_value = 0
                                day_data.append((forecast, true_value))
                                # ipdb.set_trace()

                    year = name.split('-')[-2].split('_')[-1][:4]
                    # assert root.split('/')[-1] == head[-1].split('|')[-1][3:-1]
                    company_dict[year] = np.array(year_lst)

            if company_dict:
                key = root.split('/')[-2] + '_' + root.split('/')[-1]
                self.files[key] = company_dict

        if test_keys is None:
            self.testing_keys = [('germany_TenneT_GER', '2015'),
                                 ('germany_Amprion', '2018'),
                                 ('austria_CTA', '2017'),
                                 ('belgium_CTA', '2016')]
        else:
            self.testing_keys = test_keys
        self.training_keys = []
        for key in self.files.keys():
            for year in ['2018', '2017', '2015', '2016']:
                if (key, year) not in self.testing_keys:
                    self.training_keys.append((key, year))
        self.mean, self.std = self.compute_mean_and_std()

    def get_train_complete(self, debug=False):
        complete_data = []

        for key_pair in self.training_keys:
            try:
                current_year = self.files[key_pair[0]][key_pair[1]]
                complete_data.append(current_year)
            except KeyError:
                if debug:
                    print('missing', key_pair)
                else:
                    pass
        return complete_data

    def compute_mean_and_std(self):
        complete_data = self.get_train_complete()
        gt_mean = np.mean(np.concatenate(complete_data, 0)[:, :, 1].flatten())
        gt_std = np.sqrt(np.var(np.concatenate(complete_data, 0)[:, :, 1].flatten()))
        return gt_mean, gt_std

    def _get_set(self, keys, shuffle=False, debug=False):
        batch_lst = []
        for key_pair in keys:
            try:
                current_year = self.files[key_pair[0]][key_pair[1]]
                for i in range(0, current_year.shape[0]-self.context):
                    batch_lst.append(current_year[i:i+self.context, :, :])
            except KeyError:
                if debug:
                    print('missing', key_pair)
                else:
                    pass
        # shuffle the batches.
        if shuffle:
            random.shuffle(batch_lst)
        return batch_lst

    def get_training_set(self):
        return self._get_set(keys=self.training_keys, shuffle=True)

    def get_test_set(self):
        return self._get_set(keys=self.testing_keys, shuffle=False)


class MergePowerHandler(PowerDataHandler):
    """A power handler consiting of multiple power handlers using differing
       sampling rates."""

    def __init__(self, context, power_handler_lst, testing_keys):
        self.context = context
        self._samples_per_day = np.min([ph._samples_per_day for ph in power_handler_lst])
        self.testing_keys = testing_keys
        self.files = {}

        for ph in power_handler_lst:
            for key in ph.files.keys():
                if ph._samples_per_day == self._samples_per_day:
                    self.files[key] = ph.files[key]
                else:
                    years = {}
                    for year, array in ph.files[key].items():
                        array_skip = array[:, 0::(ph._samples_per_day//24), :]
                        years[year] = array_skip
                    self.files[key] = years

        self.training_keys = []
        for key in self.files.keys():
            for year in ['2018', '2017', '2015', '2016']:
                if (key, year) not in self.testing_keys:
                    self.training_keys.append((key, year))

        self.mean, self.std = self.compute_mean_and_std()


if __name__ == "__main__":
    if False:
        import matplotlib.pyplot as plt
        path = './power_data/15m_by_country_by_company/'
        power_handler = PowerDataHandler(path, samples_per_day=96)
        train_set = power_handler.get_training_set()

        key_pair = power_handler.training_keys[0]
        print(key_pair)
        year_complete = power_handler.files[key_pair[0]][key_pair[1]]
        year_complete = year_complete[:, :, 1].flatten()
        days = 15

        l, c, line, b = plt.acorr(year_complete[(96*7):(96*7)*12].astype(np.float32),
                                  maxlags=96*days)
        plt.show()
        sample_distance = (1.0/(24*60/15))
        x1 = np.arange(0, days, sample_distance)
        x = np.concatenate([-np.flip(x1, 0), [0], x1])
        plt.plot(x, c)
        import matplotlib2tikz as tikz
        tikz.save('power_autocorr.tex')
        plt.cla()
        plt.clf()

        x = np.arange(0, 7, sample_distance)
        plt.plot(x, year_complete[(96*7):(96*7)*2])
        tikz.save('power_tennet_janw2.tex')

    if False:
        path = './power_data/1h_by_country_by_company/'
        test_keys = [('france_CTA', '2015'),
                     ('france_CTA', '2018')]
        power_handler = PowerDataHandler(path, samples_per_day=24)
        train_set = power_handler.get_training_set()
    if False:
        context_days = 15
        path = './power_data/15m_by_country_by_company/'
        power_handler_min15 = PowerDataHandler(path, context_days, samples_per_day=96,
                                               test_keys={})
        path = './power_data/30m_by_country_by_company/'
        power_handler_min30 = PowerDataHandler(path, context_days, samples_per_day=48,
                                               test_keys={})
        path = './power_data/1h_by_country_by_company/'
        power_handler_1h = PowerDataHandler(path, context_days, samples_per_day=24,
                                            test_keys={})
        testing_keys = [('germany_TenneT_GER', '2015'),
                        ('germany_Amprion', '2018'),
                        ('austria_CTA', '2017'),
                        ('belgium_CTA', '2016'),
                        ('UK_nationalGrid', '2015')]
        power_handler = MergePowerHandler(context_days, [power_handler_1h,
                                                         power_handler_min30,
                                                         power_handler_min15],
                                          testing_keys=testing_keys)
        train_set = power_handler.get_training_set()
    if True:
        import numpy as np
        import scipy.signal as scisig
        import tensorflow as tf
        tf.enable_eager_execution()
        import matplotlib.pyplot as plt
        context_days = 15
        # play with compression on the power data set.
        import sys
        sys.path.insert(0, "../")
        import eager_STFT as STFT
        path = './power_data/1h_by_country_by_company/'
        power_handler_1h = PowerDataHandler(path, context_days, samples_per_day=24,
                                            test_keys={})
        test_data = power_handler_1h.get_train_complete()[0][:, :, 1].flatten()
        window_size = 128
        overlap = int(window_size*0.5)
        window = tf.constant(scisig.get_window('hann', window_size),
                             dtype=tf.float32)
        test_data = np.expand_dims(np.expand_dims(test_data, 0), 0)
        test_data = tf.constant(test_data, tf.float32)
        result_tf = STFT.stft(test_data, window, window_size, overlap)

        # low pass keep only half of the freqs.
        compression_level = 2.5
        mask_shape = result_tf.shape
        print('remaining coefficients', int(int(mask_shape[-1])/compression_level))
        mask = tf.concat([tf.ones(int(int(mask_shape[-1])/compression_level),
                                  tf.float32),
                          tf.zeros(int(mask_shape[-1]) -
                                   int(int(mask_shape[-1])/compression_level))], -1)
        scaled = STFT.istft(result_tf,
                            window,
                            nperseg=window_size,
                            noverlap=overlap)
        compressed_spec = result_tf*tf.complex(mask, 0.0)
        compressed = STFT.istft(compressed_spec,
                                window,
                                nperseg=window_size,
                                noverlap=overlap)
        print(mask)
        plt.plot(scaled.numpy()[0, 0, :])
        plt.plot(compressed.numpy()[0, 0, :])
        plt.show()
        plt.imshow(np.abs(result_tf[0, 0, :, :].numpy()))
        plt.show()
        plt.imshow(np.log(np.abs(result_tf[0, 0, :, :].numpy())))
        plt.show()

        if 0:
            # keep only half of the freqs average.
            stride = (1, 1, 2, 1)
            # filter_hight, filter_width, in_channels, depth
            rkernel = 1.*tf.ones([1, 1, 1, 1], dtype=tf.float32)
            ikernel = tf.zeros([1, 1, 1, 1], dtype=tf.float32)
            result_tf_btfc = tf.transpose(result_tf, [0, 2, 3, 1])
            x = tf.real(result_tf_btfc)
            y = tf.imag(result_tf_btfc)

            cat_x = tf.concat([x, y], axis=-1)
            cat_kernel_4_real = tf.concat([rkernel, -ikernel], axis=-2)
            cat_kernel_4_imag = tf.concat([ikernel, rkernel], axis=-2)
            cat_kernels_4_complex = tf.concat([cat_kernel_4_real,
                                               cat_kernel_4_imag],
                                              axis=-1)

            conv = tf.nn.conv2d(input=cat_x, filter=cat_kernels_4_complex,
                                strides=stride, padding='SAME')
            conv_2 = tf.split(conv, axis=-1, num_or_size_splits=2)
            res = tf.complex(conv_2[0], conv_2[1])

            # upsample to see what was lost.
            cup2d = cconv.ComplexUpSampling2D(size=(1, 2),
                                              interpolation='bilinear')
            res_up = cup2d(res)
            toistft = tf.transpose(res_up, [0, 3, 1, 2])
            mean_time = STFT.istft(toistft,
                                   window,
                                   nperseg=window_size,
                                   noverlap=overlap)
            plt.plot(scaled.numpy()[0, 0, :])
            plt.plot(mean_time.numpy()[0, 0, :])
            plt.show()
