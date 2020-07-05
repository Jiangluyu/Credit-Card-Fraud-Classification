import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data import Data


class DataProcess:
    __raw_data_class = None
    __data = None

    def __init__(self, raw_data: Data):
        self.__raw_data_class = raw_data

    def standardize_amount(self):
        self.__data = self.__raw_data_class.get_data()
        self.__data['StdAmount'] = StandardScaler().fit_transform(
            self.__data['Amount'].values.reshape(-1, 1)
        )
        self.__data = self.__data.drop(['Time', 'Amount'], axis=1)
        self.__data = self.__data.drop(['V8', 'V13', 'V15', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'], axis=1)

    def under_sampling(self):
        X = self.__data.loc[:, self.__data.columns != 'Class']
        y = self.__data.loc[:, self.__data.columns == 'Class']

        # count number of fraud records
        fraud_count = len(self.__data[self.__data.Class == 1])

        # indices of fraud & normal records
        fraud_indices = np.array(self.__data[self.__data.Class == 1].index)
        normal_indices_list = self.__data[self.__data.Class == 0].index

        # select fraud-count samples from normal records
        random_normal_indices = np.array(np.random.choice(normal_indices_list, fraud_count, replace=False))

        # concat
        under_sampling_indices = np.concatenate([fraud_indices, random_normal_indices])

        # get the records
        under_sampling_data = self.__data.iloc[under_sampling_indices, :]
        X_under_sampling = under_sampling_data.iloc[:, under_sampling_data.columns != 'Class']
        y_under_sampling = under_sampling_data.iloc[:, under_sampling_data.columns == 'Class']

        return X, y, X_under_sampling, y_under_sampling

    def cross_validation_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        return X_train, X_test, y_train, y_test
