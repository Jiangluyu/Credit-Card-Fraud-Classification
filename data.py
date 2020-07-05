import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns


class Data:
    __data = None

    def __init__(self):
        self.__data = pd.read_csv('alipay_huabei_FS1.csv')

    def get_shape(self):
        return self.__data.shape

    def show_sns_time_amount(self):
        hour_data = self.__data
        hour_data['Time'] = round(hour_data['Time'] / 3600)
        sns.lineplot(data=hour_data,
                     x=hour_data['Time'],
                     y=hour_data['Amount'],
                     hue=hour_data['Class'],
                     palette="tab10",
                     linewidth=2.5)
        plt.show()

    def show_sns_amount_distribution(self):
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 4))
        bins = 30
        ax1.hist(self.__data[self.__data.Class == 1]['Amount'], bins=bins)
        ax1.set_title('Fraud')
        ax2.hist(self.__data[self.__data.Class == 0]['Amount'], bins=bins)
        ax2.set_title('Normal')
        plt.xlabel('Amount')
        plt.ylabel('Number of Transactions')
        plt.yscale('log')
        plt.show()

    def show_sns_time_distribution(self):
        plt.figure(figsize=[16, 4])
        hour_data = self.__data
        hour_data['Time'] = round(hour_data['Time'] / 3600)
        sns.distplot(hour_data[hour_data['Class'] == 1]['Time'], bins=50)
        sns.distplot(hour_data[hour_data['Class'] == 0]['Time'], bins=100)
        plt.show()

    def show_sns_V_distribution(self):
        plt.figure(figsize=(12, 28 * 4))
        v_features = self.__data.iloc[:, 1:29].columns
        gs = gridspec.GridSpec(28, 1)
        for i, v_feature_num in enumerate(self.__data[v_features]):
            ax = plt.subplot(gs[i])
            sns.distplot(self.__data[self.__data.Class == 1][v_feature_num], bins=50)
            sns.distplot(self.__data[self.__data.Class == 0][v_feature_num], bins=50)
            ax.set_xlabel('')
            ax.set_title(v_feature_num)
        plt.show()

    def get_class_values(self):
        return pd.value_counts(self.__data['Class'])

    def get_data(self):
        return self.__data