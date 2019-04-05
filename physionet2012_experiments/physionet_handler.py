import csv
import os
import datetime
import numpy as np
import random

from IPython.core.debugger import Tracer
debug_here = Tracer()


class PhysioHandler(object):
    '''
    Data handler meant to load and preprocess
    power load data from:
    https://physionet.org/challenge/2012/
    '''

    def __init__(self, set_path='./data/set-a/',
                 label_file='set-a_outcome.txt',
                 cut_at=None):
        '''
        Creates a power data handler.
        Arguments:
            set-path: The path to the physionet challenge data points.
            label_file: The location of the label file.
            cut_at: Maximum number of matrix colums.
        Returns:
            physio dict: A dictionary with the physionet samples.
        '''
        self.path = set_path

        # walk trough subfolders of the path and find csv files.
        self.files = {}
        self.data = {}
        for root, dirs, files in os.walk(self.path):
            # print(root, dirs)
            patient_dict = {}
            for file_name in files:
                with open(root + file_name, newline='') as csvfile:
                    med_reader = csv.reader(csvfile, delimiter=',')
                    patient_value_dict = {}
                    patient_value_dict['TimePoints'] = []
                    for i, row in enumerate(med_reader):
                        if i == 0:
                            # rows have values time, key, value (in that order)
                            assert row[0] == 'Time'
                            assert row[1] == 'Parameter'
                            assert row[2] == 'Value'
                        else:
                            key_lst = list(patient_value_dict.keys())
                            current_key = row[1]
                            if current_key not in key_lst:
                                new_key_lst = [(row[0], row[2])]
                                patient_value_dict[current_key] = new_key_lst
                            else:
                                patient_value_dict[current_key].append(
                                    (row[0], row[2]))
                            if row[0] not in patient_value_dict['TimePoints']:
                                patient_value_dict['TimePoints'].append(row[0])
                file_key = int(file_name.split('.')[0])
                patient_dict[file_key] = patient_value_dict
        self.patient_dict = patient_dict
        self.key_lst = list(self.patient_dict.keys())
        self.matrix_dict = self.scale_and_merge()
        self.max_length = np.max(self.lengths())
        self.mean_dict, self.std_dict = self.compute_mean_and_std()

        # load the labels.
        with open(label_file, newline='') as csvfile:
            med_label_reader = csv.reader(csvfile, delimiter=',')
            patient_label_dict = {}
            for i, row in enumerate(med_label_reader):
                if i == 0:
                    # rows have values time, key, value (in that order)
                    row[0] == 'RecordID'
                    row[1] == 'SAPS-I'
                    row[2] == 'SOFA'
                    row[3] == 'Length_of_stay'
                    row[4] == 'Survival'
                    row[5] == 'In-hospital_death'
                else:
                    current_patient_label_dict = {}
                    current_patient_label_dict['SAPS-I'] = int(row[1])
                    current_patient_label_dict['SOFA'] = int(row[2])
                    current_patient_label_dict['Length_of_stay'] = int(row[3])
                    current_patient_label_dict['Survival'] = int(row[4])
                    current_patient_label_dict['In-hospital_death'] = int(row[5])
                    patient_label_dict[int(row[0])] = current_patient_label_dict
        self.patient_label_dict = patient_label_dict
        self.epoch_lst = self.generate_epoch_lst(cut_at=cut_at)

    @property
    def recorded_quantities(self):
        return ['TimePoints', 'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin',
                'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol', 'Creatinine',
                'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR',
                'K', 'Lactate', 'Mg', 'MAP', 'MechVent', 'Na', 'NIDiasABP',
                'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
                'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC']

    def lengths(self):
        len_lst = []
        # the length values
        for key, current_dict in self.patient_dict.items():
            len_lst.append(len(current_dict['TimePoints']))
        # the data-set mean values.
        return len_lst

    def compute_mean_and_std(self):
        val_dict = {}
        for _, record in self.patient_dict.items():
            for key, val in record.items():
                # print(key)
                if key == 'TimePoints':
                    val = [self.time_string_to_seconds(v)
                           for v in val]
                else:
                    if (len(val)) > 1:
                        val = [float(v[1]) for v in val]
                    else:
                        val = [float(val[0][1])]
                try:
                    val_dict[key].extend(val)
                except KeyError:
                    if type(val) is list:
                        val_dict[key] = val
                    else:
                        val_dict[key] = [val]
        mean_dict = {}
        std_dict = {}
        for key, val in val_dict.items():
            mean_dict[key] = np.mean(val)
            std_dict[key] = np.std(val)
            if std_dict[key] < 0.001:
                std_dict[key] = 1.0
        return mean_dict, std_dict

    def time_string_to_seconds(self, time_string):
        '''
        Convert a time string 'hh:mm' to seconds.
        '''
        hours, minutes = time_string.split(':')
        hours = int(hours)
        minutes = int(minutes)
        seconds = hours*60*60 + minutes*60
        return seconds

    def format_data(self, data_dict, fill_value=np.NaN):
        '''
        Turn an unstructured data value dictionary into a padded array.
        '''

        def compare_and_fill(time_measurement_list, time_second_lst, fill_value):
            '''
            Compare available measurements and mark missing values with a fill value.
            '''
            format_list = []
            for time, value in time_measurement_list:
                time_s = self.time_string_to_seconds(time)
                value = float(value)
                format_list.append((time_s, value))

            full_format_list = []
            for time_point in time_second_lst:
                hit = False
                for measurement_time, measurement in format_list:
                    if time_point == measurement_time:
                        # found a valid measurement
                        full_format_list.append(measurement)
                        hit = True
                        # sometimes people measure two values
                        # lets ignore the second one.
                        break
                if hit is False:
                    # found nothing fill
                    full_format_list.append(fill_value)
            return full_format_list

        format_dict = {}
        time_points = len(data_dict['TimePoints'])
        for key, data_list in data_dict.items():
            if key == 'TimePoints':
                time_second_lst = []
                for time in data_list:
                    time_second_lst.append(self.time_string_to_seconds(time))
                format_dict[key] = np.array(time_second_lst)
            elif key == 'Age':
                age = [int(data_list[0][1])]*time_points
                format_dict[key] = np.array(age)
            elif key == 'Gender':
                gender = [int(data_list[0][1])]*time_points
                format_dict[key] = np.array(gender)
            elif key == 'Height':
                height = [float(data_list[0][1])]*time_points
                format_dict[key] = np.array(height)
            elif key == 'ICUType':
                icu_type = [int(data_list[0][1])]*time_points
                format_dict[key] = np.array(icu_type)
            elif key == 'Weight':
                weight = [float(data_list[0][1])]*time_points
                format_dict[key] = np.array(weight)
            elif key == 'RecordID':
                pass
            else:
                formatted = compare_and_fill(data_list, time_second_lst,
                                             fill_value)
                if len(formatted) != time_points:
                    debug_here()
                assert len(formatted) == time_points
                format_dict[key] = np.array(formatted)

        return format_dict

    def scale_and_merge(self):
        '''
        Scale the data mean subtraction and std division and place
        and merge data into an array.
        '''
        formatted_data = {}
        max_dict = {}
        # find the maximum values along every dimension.
        for file_key, data in self.patient_dict.items():
            formatted_data[file_key] = self.format_data(data)
            for key, array in formatted_data[file_key].items():
                if key not in max_dict.keys():
                    max_dict[key] = 0

                max_val = np.nanmax(array)
                if max_val > max_dict[key]:
                    max_dict[key] = max_val

        # scale by maximum value.
        scaled_data = {}
        for file_key, data_dict in formatted_data.items():
            # print(file_key)
            scaled_array = None
            # for data_key, array in data_dict.items():
            for data_key in self.recorded_quantities:
                try:
                    array = data_dict[data_key]
                    scaled_row = array/max_dict[data_key]
                except KeyError:
                    scaled_row = np.array([np.NaN]*len(data_dict['TimePoints']))

                if scaled_array is None:
                    scaled_array = np.expand_dims(scaled_row, 0)
                else:
                    scaled_row = np.expand_dims(scaled_row, 0)
                    scaled_array = np.concatenate([scaled_array, scaled_row], 0)

            scaled_data[file_key] = scaled_array
        return scaled_data

    def generate_epoch_lst(self, cut_at=None):
        epoch = []
        for key in self.key_lst:
            mat = self.matrix_dict[key]
            mat = np.nan_to_num(mat)
            if cut_at:
                if mat.shape[1] > cut_at:
                    start = mat.shape[1] - cut_at
                    mat = mat[:, start:]
                to_pad = cut_at - mat.shape[1]
            else:
                to_pad = self.max_length - mat.shape[1]
            padded_mat = np.pad(mat, [[0, 0], [0, to_pad]], mode='constant')
            label_array = np.array(self.patient_label_dict[key]['In-hospital_death'])
            epoch.append((padded_mat, label_array, key))
        return epoch

    def get_batches(self, batch_size):
        '''
        pad matrices and return a batch.
        '''
        assert len(self.patient_dict) % batch_size == 0, 'batch_size invalid'
        epoch = self.epoch_lst
        random.shuffle(epoch)
        data_lst = []
        label_lst = []
        for el in epoch:
            data_lst.append(el[0])
            label_lst.append(el[1])
        data_array = np.array(data_lst)
        data_batches = np.split(data_array, 4000//batch_size, axis=0)
        label_array = np.array(label_lst)
        label_batches = np.split(label_array, 4000//batch_size, axis=0)
        return data_batches, label_batches


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.signal as scisig
    set_path = './data/set-a/'
    label_file = './data/set-a_outcome.txt'
    physio_handler = PhysioHandler(set_path, label_file)
    batch = physio_handler.get_batches(25)
    len_lst = physio_handler.lengths()
    # plt.hist(len_lst)
    # plt.show()
    # example_mat = batch[0][0][0, :, :]
    # plt.imshow(example_mat)
    # plt.show()
    # spec = np.abs(scisig.stft(example_mat, nperseg=12))[-1]
    # plt.imshow(np.concatenate(np.squeeze(np.split(spec, spec.shape[0], axis=0))))
    # plt.show()
