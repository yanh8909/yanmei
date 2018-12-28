import json
from pandas import read_csv, DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time
import copy
import numpy as np

time_format = "%Y-%m-%d %H:%M:%S.%f"


def read_data(filename: str, config_file: str) -> DataFrame:              # 封装读取数据函数
    """
    :param filename: 数据文件
    :param config_file: 配置文件（json）
    :return:

    """
    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)
    str_columns_list = {column: str for column in config.get("str_columns")}
    encoding = config.get("encoding")
    separator = config.get("separator")
    na_values_dict = config.get("na_value")
    data = read_csv(filename, encoding=encoding, na_values=na_values_dict, dtype=str_columns_list, sep=separator)
    return data


def get_columns(config_file: str):                          # 封装获取列函数
    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)
    bin_columns = config.get("bin_columns")
    con_columns = config.get("con_columns")
    cat_columns = config.get("cat_columns")
    target_column = config.get("target_column")
    return bin_columns, con_columns, cat_columns, target_column


def get_useless(config_file: str):
    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)
    useless_columns = config.get("useless_columns")
    return useless_columns


def get_day_diff(from_time: str, target_time: str):
    if not isinstance(from_time, str) or not isinstance(target_time, str):
        return 0
    if len(from_time) == 0 or len(target_time) == 0:
        return 0
    from_time_tup = time.mktime(time.strptime(from_time, time_format))
    target_time_tup = time.mktime(time.strptime(target_time, time_format))
    time_diff = target_time_tup - from_time_tup
    return int(time_diff/60/60/24)


def get_year_diff(from_time: str, target_time: str):
    return float(get_day_diff(from_time, target_time)/365.25)


def bin_column_analysis(bin_column: Series):
    print(bin_column.describe())
    print("There are "+bin_column.isnull().sum()+" nan values")
    print("Null values accounts for  %2f".format(bin_column.isnull().sum()/len(bin_column)))
    print("In this column, there are ", bin_column.unique(), " items")


def con_column_analysis(con_column: Series):
    print(con_column.describe())
    print("There are "+con_column.isnull().sum()+" nan values.")
    print("Null values accounts for  %2f".format(con_column.isnull().sum()/len(con_column)))
    print("This column has " + con_column.unique().__len__() + "items.")


def cat_column_analysis(cat_column: Series):
    print(cat_column.describe())
    print("There are " + cat_column.isnull().sum() + " nan values")
    print("Null values accounts for  %2f".format(cat_column.isnull().sum()/len(cat_column)))
    print("In this column, there are ", cat_column.unique(), " items")


def print_columns_analysis(data: DataFrame, columns: list, analysis_func):
    for i in columns:
        print("===========Analysis of "+i)
        analysis_func(data[i])
        print("==============================")


def train_test_split_by_time(data: DataFrame, time_column: str=None, split_date=None, train_size=None, test_size=None):
    if time_column is not None:
        test_set = data[data[time_column] >= split_date]
        train_set = data[data[time_column] < split_date]
    elif train_size is not None:
        train_set, test_set = train_test_split(data, train_size=train_size)
    elif test_size is not None:
        train_set, test_set = train_test_split(data, train_size=train_size)
    else:
        print("pleas set one parameter at last!")
        train_set, test_set = None, None
    return train_set, test_set


def save_train_test_sets(data: DataFrame, train_file_name: str, test_file_name: str, save_mode: str="pickle", **kwargs):
    train_set, test_set = train_test_split_by_time(data, **kwargs)
    if save_mode == "csv":
        train_set.to_csv(train_file_name, header=True, index=False)
        test_set.to_csv(test_file_name, header=False, index=False)
    elif save_mode == "pickle":
        train_set.to_pickle(train_file_name)
        test_set.to_pickle(test_file_name)
    else:
        print("please choose right save mode")


class LabelsEncoder:
    def __init__(self):
        self.labels_encoder = None
        self.columns = None

    def fit(self, data: DataFrame):
        self.columns = data.columns
        self.labels_encoder = {i: LabelEncoder() for i in self.columns}
        [self.labels_encoder.get(i).fit(data[i]) for i in self.columns]

    def transform(self, data: DataFrame, columns_method: str= "intersection"):
        fit_columns = self.columns
        transform_columns = data.columns
        if columns_method == "intersection":
            labels_enoded_list = [self.labels_encoder.get(col_name).transform(data[col_name]) for col_name in transform_columns if col_name in fit_columns]
            return np.array(labels_enoded_list).T
        elif columns_method == "raise":
            fit_columns_for_judge = [col for col in fit_columns if col in transform_columns]
            if len(fit_columns) == len(fit_columns_for_judge):
                labels_enoded_list = [encoder.transform(data[name]) for name, encoder in self.labels_encoder]
                return np.array(labels_enoded_list).T
            else:
                print("columns length are not equel")
        else:
            print("function not completed")

    def fit_transform(self, data: DataFrame, columns_method: str="intersection"):
        self.fit(data)
        return self.transform(data, columns_method)


class DataProcesser:
    def __init__(self, data_filename, config_file):
        self.data_filename = data_filename
        self.config_file = config_file
        self.data = read_data(data_filename, config_file)
        self.raw_data = copy.deepcopy(self.data)
        self.length = len(self.data)
        self.bin_columns, self.con_columns, self.cat_columns, self.target_column = get_columns(config_file)

    def preprocess_columns(self, na_value_filter=1.0, bin_filter=None, con_filter=None, cat_filter=None):
        pass

    def __preprocess_columns__(self, column_type, filter_level, fill_value="auto"):
        if column_type not in ["bin","cat","con"]:
            raise ValueError("{} is not in bin, cat or con".format(column_type))
        if column_type == "bin":
            column_processed = []
            for i in self.bin_columns:
                if self.data[i].isnull().sum/self.length > filter_level:
                    del self.data[i]
                    continue
                desc = self.data[i].describe()
                if desc.get("unique") == 1:
                    del self.data[i]
                else:
                    if fill_value == "auto":
                        self.data[i] = self.data[i].fillna(desc.get("top"))
                    else:
                        self.data[i] = self.data[i].fillna(fill_value)
                    column_processed.append(i)
        elif column_type == "con":
            column_processed = []
            pass
        elif column_type == "cat":
            column_processed = []
            pass
        else:
            column_processed = []
        return column_processed


if __name__ == "__main__":
    # test read_data()
    test_data = read_data("test_data_1000.csv", "features.json")
    print(test_data)

    # test get_columns()
    bin_columns, con_columns, cat_columns, target_column = get_columns("features.json")
    print(bin_columns, con_columns, cat_columns, target_column)

    # test get_day_diff() & get_year_diff()
    test_from_date = "2012-01-01 20:20:12.000"
    test_end_date = "2013-01-01 20:27:22.000"
    print(get_day_diff(test_from_date, test_end_date))
    print(get_year_diff(test_from_date, test_end_date))

    # test save_train_test_sets()
    save_train_test_sets(test_data, "train_data.pkl", "test_data.pkl", train_size=0.8)



