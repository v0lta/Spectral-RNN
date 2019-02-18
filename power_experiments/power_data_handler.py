import csv
import os
import datetime
import numpy as np
import random
import ipdb
dbg = ipdb.set_trace


def exit():
    os._exit(1)


class PowerDataHandler(object):
    '''
    Data handler meant to load and preprocess
    power load data from:
    https://transparency.entsoe.eu/
    '''
    def __init__(self, path, context=15):
        '''
        Creates a power data handler.

        '''
        self.path = path
        self.context = context

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
                                year_lst.append(day_data)
                                day_data = []
                                # ipdb.set_trace()
                                prev_day = current_day
                            if i == 0:
                                # TODO assertions.
                                # head = row
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

                    # TODO: check head append to year list!
                    year = name.split('-')[-2].split('_')[-1][:4]
                    # assert root.split('/')[-1] == head[-1].split('|')[-1][3:-1]
                    company_dict[year] = np.array(year_lst)
            if company_dict:
                key = root.split('/')[-2] + '_' + root.split('/')[-1]
                self.files[key] = company_dict

        self.testing_keys = [('germany_TenneT_GER', '2015'),
                             ('germany_Amprion', '2018'),
                             ('austria_CTA', '2017'),
                             ('belgium_CTA', '2016')]

        self.training_keys = []
        for key in self.files.keys():
            for year in ['2018', '2017', '2015', '2016']:
                if (key, year) not in self.testing_keys:
                    self.training_keys.append((key, year))

        self.mean, self.std = self.compute_mean_and_std()

    def compute_mean_and_std(self):
        complete_data = []
        for key_pair in self.training_keys:
            current_year = self.files[key_pair[0]][key_pair[1]]
            complete_data.append(current_year)
        gt_mean = np.mean(np.concatenate(complete_data, 0)[:, :, 1].flatten())
        gt_std = np.sqrt(np.var(np.concatenate(complete_data, 0)[:, :, 1].flatten()))
        return gt_mean, gt_std

    def _get_set(self, keys, shuffle=False):
        batch_lst = []
        for key_pair in keys:
            current_year = self.files[key_pair[0]][key_pair[1]]
            for i in range(0, current_year.shape[0]-self.context):
                batch_lst.append(current_year[i:i+self.context, :, :])
        # shuffle the batches.
        if shuffle:
            random.shuffle(batch_lst)
        return batch_lst

    def get_training_set(self):
        return self._get_set(keys=self.training_keys, shuffle=True)

    def get_test_set(self):
        return self._get_set(keys=self.testing_keys, shuffle=False)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    path = './power_data/by_country_by_company/'
    power_handler = PowerDataHandler(path)
    train_set = power_handler.get_training_set()
