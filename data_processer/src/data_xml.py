#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import xml.etree.ElementTree as ET
import os
import sys
import time

time_format = "%Y-%m-%d %H:%M:%S.%f"

def bin_column_analysis(bin_column: pd.Series):
    """
    用于分析二分类特征的空值数量、占比情况和取值情况
    :param bin_column:
    :return:
    """
    print(bin_column.describe())
    print("There are "+bin_column.isnull().sum()+" nan values")
    print("Null values accounts for  %2f".format(bin_column.isnull().sum()/len(bin_column)))
    print("In this column, there are ", bin_column.unique(), " items")


def con_column_analysis(con_column: pd.Series):
    """
    用于分析连续值特征的空值数量、占比情况和取值情况
    :param con_column:
    :return:
    """
    print(con_column.describe())
    print("There are "+con_column.isnull().sum()+" nan values.")
    print("Null values accounts for  %2f".format(con_column.isnull().sum()/len(con_column)))
    print("This column has " + con_column.unique().__len__() + "items.")


def cat_column_analysis(cat_column: pd.Series):
    """
    用于分析离散值特征的空值数量、占比情况和取值情况
    :param cat_column:
    :return:
    """
    print(cat_column.describe())
    print("There are " + cat_column.isnull().sum() + " nan values")
    print("Null values accounts for  %2f".format(cat_column.isnull().sum()/len(cat_column)))
    print("In this column, there are ", cat_column.unique(), " items")


def print_columns_analysis(data: pd.DataFrame, columns: list, analysis_func):
    """
    用于将描述性函数应用给每一列函数
    :param data:
    :param columns:
    :param analysis_func:
    :return:
    """
    for i in columns:
        print("===========Analysis of "+i)
        analysis_func(data[i])
        print("==============================")


def train_test_split_by_time(data: pd.DataFrame, time_column: str=None, split_date=None, train_size=None, test_size=None):
    """
    not tested, do not use
    :param data:
    :param time_column:
    :param split_date:
    :param train_size:
    :param test_size:
    :return:
    """
    from sklearn.model_selection import train_test_split
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


def save_train_test_sets(data: pd.DataFrame, train_file_name: str, test_file_name: str, save_mode: str="pickle", **kwargs):
    """
    not tested, do not use
    :param data:
    :param train_file_name:
    :param test_file_name:
    :param save_mode:
    :param kwargs:
    :return:
    """
    train_set, test_set = train_test_split_by_time(data, **kwargs)
    if save_mode == "csv":
        train_set.to_csv(train_file_name, header=True, index=False)
        test_set.to_csv(test_file_name, header=False, index=False)
    elif save_mode == "pickle":
        train_set.to_pickle(train_file_name)
        test_set.to_pickle(test_file_name)
    else:
        print("please choose right save mode")


def get_day_diff(start_data_series: pd.Series, end_data_series: pd.Series):
    """
    :param start_data_series: 起始时间列
    :param end_data_series: 结束时间列
    :return:
    """
    diff = pd.to_datetime(end_data_series) - pd.to_datetime(start_data_series)
    return pd.to_numeric(diff)/3600/24/1000000000


def get_year_diff(start_data_series: pd.Series, end_data_series: pd.Series):
    """
    :param start_data_series: 起始时间列
    :param end_data_series: 结束时间列
    :return:
    """
    return get_day_diff(start_data_series, end_data_series)/365.25


def read_xml(config_file):  # 读取xml文件，并返回根元素
    """
    用于读取xml文件的函数
    :param config_file:
    :return:
    """
    xmlFilePath = os.path.abspath(config_file)

    try:
        tree = ET.parse(xmlFilePath)
        root = tree.getroot()
        return root
    except Exception as e:
        print("Parse xml fail")
        sys.exit()


class XmlConfigParser:
    """
    解析xml文件
    The document test content
    >>> config = XmlConfigParser("features.xml")
    >>> config.get_attribute("encode")
    'utf-8'
    >>> config.get_attribute("train_features")
    ['hn.ca_bxperiod', '投保人身高']
    >>> config.get_columns_dict("dtype")
    {'e.rowkey': 'str', 'hn.ca_bxperiod': 'str', 'hn.ca_cnzs': 'str', 'hn.ca_jfperiod': 'str', 'h.c_pl196_h_investigate_flag_trace': 'str', 'h.c_in072_h_revoked_flag_trace': 'str', 'h.c_in073_h_reinforce_flag_trace': 'str', 'i.c_pl195_i_investigate_flag_trace': 'str', 'i.c_in073_i_cii_one_year_occured_flag_trace': 'str', 'i.c_in078_i_increase_premium_flag_trace': 'str', 'i.c_in075_i_special_clause_flag_trace': 'str', 'i.c_in068_i_surrendered_flag_trace': 'str', 'i.c_in069_i_revoked_flag_trace': 'str', 'i.c_in070_i_invalid_flag_trace': 'str', 'i.c_in071_i_reinforce_flag_trace': 'str', 'i.c_in076_i_abnormal_bmi_flag_trace': 'str', 'i.c_in077_i_smoke_overlimit_flag_trace': 'str', '投保人性别': 'str', '投保人异常告知标记': 'str', '被保人性别': 'str', 'NO1': 'str', 'NO2': 'str', 'NO3': 'str', 'NO4': 'str', 'NO5': 'str', 'NO6': 'str', 'NO7': 'str', 'NO8': 'str', 'NO9': 'str', 'NO10': 'str', 'NO11': 'str', 'NO12': 'str', 'NO13': 'str', 'NO14': 'str', 'NO15': 'str', 'NO16': 'str', 'NO17': 'str', 'NO18': 'str', 'NO19': 'str', 'NO20': 'str', 'NO21': 'str', 'NO22': 'str', 'NO23': 'str', 'NO24': 'str', 'NO25': 'str', 'NO26': 'str', 'NO27': 'str', 'NO28': 'str', 'NO29': 'str', 'hn.itrvl_code': 'str', 'hn.pol_code': 'str', 'hn.rel_to_hldr': 'str', 'hn.mr_type': 'str', 'hn.sales_channel_code': 'str', 'c_bs007_city_code_h': 'str', 'c_bs009_prov_code_h': 'str', 'c_bs017_marriage_stat_h': 'str', 'c_bs022_occ_class_dept_code_h': 'str', 'c_bs030_sos_relation_h': 'str', 'c_bs031_occ_class_code_h': 'str', 'c_bs032_occ_sub_code_h': 'str', 'c_bs033_occ_code_h': 'str', 'c_bs035_job_risk_h': 'str', 'income_band_h': 'str', 'manage_code_h': 'str', 'c_bs007_city_code_i': 'str', 'c_bs009_prov_code_i': 'str', 'c_bs017_marriage_stat_i': 'str', 'c_bs022_occ_class_dept_code_i': 'str', 'c_bs030_sos_relation_i': 'str', 'c_bs031_occ_class_code_i': 'str', 'c_bs032_occ_sub_code_i': 'str', 'c_bs033_occ_code_i': 'str', 'c_bs035_job_risk_i': 'str', 'id_type_i': 'str', 'income_band_i': 'str', 'manage_code_i': 'str', 'x.s_bs008_credit_level': 'str', 'x.s_bs010_did': 'str', '接入来源 ': 'str', '投保人婚姻状况': 'str', '投保人职业大类代码': 'str', '投保人职业子类代码': 'str', '投保人职业细类代码': 'str', '投保人职业风险等级 ': 'str', '投保人客户证件代码类别': 'str', '被保人婚姻状况': 'str', '被保人异常告知标记': 'str', '被保人职业大类代码': 'str', '被保人职业子类代码': 'str', '被保人职业细类代码': 'str', '被保人职业风险等级 ': 'str', '被保人客户证件代码类别': 'str', '被保人黑名单标记': 'str', '保单接入来源': 'str', '投保日期所在月份': 'str', 'hn.cntr_no': 'str', 'hn.expiry_date': 'str', 'hn.mg_branch_no': 'str', 'hn.n_sales_branch_no': 'str', 'hn.n_sales_code': 'str', 'hn.p_bs002_city_code': 'str', 'hn.p_bs004_prov_code': 'str', 'hn.p_bs022_verfication_mark': 'str', 'hn.p_bs024_udw_result': 'str', 'hn.p_pl005_pid': 'str', 'hn.cntr_stat': 'str', 'hn.p_in006_investigate_flag': 'str', 'holder_party_id_reverse': 'str', 'id_no_h': 'str', 'id_type_h': 'str', 'name_h': 'str', 'birth_h': 'str', 'insured_party_id_reverse': 'str', 'id_no_i': 'str', 'name_i': 'str', 'birth_i': 'str', 'h.c_pl215_most_pol_class2_desc_trace': 'str', 'i.pl_01_02_01_trace ': 'str', 'i.pl_01_02_02_trace': 'str', 'i.c_pl300_i_last_claim_date_trace': 'str', 'x.s_bs004_birthday': 'str', 'x.s_bs009_transfer_date': 'str', 'x.s_bs016_personid': 'str', 'x.s_bs028_inour_date': 'str', 'e.p_cp037_first_occur_date': 'str', 'e.p_cp013_reason_1': 'str', 'e.in_force_date': 'str', '合同ID': 'str', '合同号': 'str', '投保单号': 'str', '投保人生日': 'str', '投保人客户证件代码': 'str', '投保人通讯地址': 'str', '投保人邮编': 'str', '被保人生日': 'str', '被保人客户证件代码': 'str', 'CNTR_ID': 'str', 'APPL_NO': 'str', 'IPSN_CUST_NO': 'str', '投保人身高': 'float'}
    >>> config.get_columns_dict("ftype")
    {'e.rowkey': 'useless', 'hn.ca_bxperiod': 'con', 'hn.ca_cnzs': 'bin', 'hn.ca_jfperiod': 'cat', 'h.c_pl196_h_investigate_flag_trace': 'bin', 'h.c_in072_h_revoked_flag_trace': 'bin', 'h.c_in073_h_reinforce_flag_trace': 'bin', 'i.c_pl195_i_investigate_flag_trace': 'bin', 'i.c_in073_i_cii_one_year_occured_flag_trace': 'useless', 'i.c_in078_i_increase_premium_flag_trace': 'bin', 'i.c_in075_i_special_clause_flag_trace': 'bin', 'i.c_in068_i_surrendered_flag_trace': 'bin', 'i.c_in069_i_revoked_flag_trace': 'bin', 'i.c_in070_i_invalid_flag_trace': 'bin', 'i.c_in071_i_reinforce_flag_trace': 'bin', 'i.c_in076_i_abnormal_bmi_flag_trace': 'bin', 'i.c_in077_i_smoke_overlimit_flag_trace': 'bin', '投保人性别': 'bin', '投保人异常告知标记': 'bin', '被保人性别': 'bin', 'NO1': 'bin', 'NO2': 'bin', 'NO3': 'bin', 'NO4': 'bin', 'NO5': 'bin', 'NO6': 'bin', 'NO7': 'bin', 'NO8': 'bin', 'NO9': 'bin', 'NO10': 'bin', 'NO11': 'bin', 'NO12': 'bin', 'NO13': 'bin', 'NO14': 'bin', 'NO15': 'bin', 'NO16': 'bin', 'NO17': 'bin', 'NO18': 'bin', 'NO19': 'bin', 'NO20': 'bin', 'NO21': 'bin', 'NO22': 'bin', 'NO23': 'bin', 'NO24': 'bin', 'NO25': 'bin', 'NO26': 'bin', 'NO27': 'bin', 'NO28': 'bin', 'NO29': 'bin', 'hn.itrvl_code': 'cat', 'hn.pol_code': 'cat', 'hn.rel_to_hldr': 'cat', 'hn.mr_type': 'bin', 'hn.sales_channel_code': 'cat', 'c_bs007_city_code_h': 'cat', 'c_bs009_prov_code_h': 'cat', 'c_bs017_marriage_stat_h': 'useless', 'c_bs022_occ_class_dept_code_h': 'useless', 'c_bs030_sos_relation_h': 'useless', 'c_bs031_occ_class_code_h': 'useless', 'c_bs032_occ_sub_code_h': 'useless', 'c_bs033_occ_code_h': 'useless', 'c_bs035_job_risk_h': 'useless', 'income_band_h': 'cat', 'manage_code_h': 'useless', 'c_bs007_city_code_i': 'useless', 'c_bs009_prov_code_i': 'useless', 'c_bs017_marriage_stat_i': 'useless', 'c_bs022_occ_class_dept_code_i': 'useless', 'c_bs030_sos_relation_i': 'useless', 'c_bs032_occ_sub_code_i': 'useless', 'c_bs033_occ_code_i': 'useless', 'c_bs035_job_risk_i': 'useless', 'id_type_i': 'useless', 'income_band_i': 'cat', 'manage_code_i': 'useless', 'x.s_bs008_credit_level': 'cat', 'x.s_bs010_did': 'cat', '接入来源 ': 'cat', '投保人婚姻状况': 'cat', '投保人职业大类代码': 'cat', '投保人职业子类代码': 'cat', '投保人职业细类代码': 'cat', '投保人职业风险等级 ': 'cat', '投保人客户证件代码类别': 'cat', '被保人婚姻状况': 'cat', '被保人异常告知标记': 'bin', '被保人职业大类代码': 'cat', '被保人职业子类代码': 'cat', '被保人职业细类代码': 'cat', '被保人职业风险等级 ': 'cat', '被保人客户证件代码类别': 'cat', '被保人黑名单标记': 'bin', '保单接入来源': 'useless', '投保日期所在月份': 'cat', 'hn.cntr_no': 'useless', 'hn.expiry_date': 'useless', 'hn.mg_branch_no': 'useless', 'hn.n_sales_branch_no': 'useless', 'hn.n_sales_code': 'useless', 'hn.p_bs002_city_code': 'useless', 'hn.p_bs004_prov_code': 'useless', 'hn.p_bs022_verfication_mark': 'useless', 'hn.p_bs024_udw_result': 'useless', 'hn.p_pl005_pid': 'useless', 'hn.cntr_stat': 'useless', 'hn.p_in006_investigate_flag': 'useless', 'holder_party_id_reverse': 'useless', 'id_no_h': 'useless', 'id_type_h': 'useless', 'name_h': 'useless', 'birth_h': 'useless', 'insured_party_id_reverse': 'useless', 'id_no_i': 'useless', 'name_i': 'useless', 'birth_i': 'useless', 'h.c_pl215_most_pol_class2_desc_trace': 'useless', 'i.pl_01_02_01_trace ': 'cat', 'i.pl_01_02_02_trace': 'cat', 'i.c_pl300_i_last_claim_date_trace': 'useless', 'x.s_bs004_birthday': 'useless', 'x.s_bs009_transfer_date': 'useless', 'x.s_bs016_personid': 'useless', 'x.s_bs028_inour_date': 'useless', 'e.p_cp037_first_occur_date': 'useless', 'e.p_cp013_reason_1': 'useless', 'e.in_force_date': 'useless', '合同ID': 'useless', '合同号': 'useless', '投保单号': 'useless', '投保人生日': 'useless', '投保人客户证件代码': 'useless', '投保人通讯地址': 'useless', '投保人邮编': 'useless', '被保人生日': 'useless', '被保人客户证件代码': 'useless', 'CNTR_ID': 'useless', 'APPL_NO': 'useless', 'x.s_bs019_bid': 'useless', 'sex_h': 'bin', 'sex_i': 'useless', '被保人同业公司人身险保额合计': 'useless', 'IPSN_CUST_NO': 'useless', 'income_h': 'useless', 'income_i': 'useless', 'illness.compensate_way ': 'cat', 'hn.ca_sumprem ': 'con', 'hn.ca_yearprem': 'con', 'hn.s_pl001_unitssold': 'con', 'hn.s_pl002_standardunitssold': 'con', 'hn.s_pl003_rescuedtimes': 'con', 'hn.s_pl005_salesstandardratio': 'con', 'hn.s_pl006_salestwoyearsoccurrate': 'con', 'hn.ca_amount': 'con', 'h.c_pl002_h_l_exp_count_trace': 'con', 'h.c_pl019_h_count_trace': 'con', 'h.c_pl061_h_heal_count_trace': 'con', 'h.c_pl062_h_heal_count_self_trace': 'con', 'h.c_pl084_cancel_plcy_lst1y_trace': 'con', 'h.c_pl096_comp_quit_lst5y_trace': 'con', 'h.c_pl003_max_diff_trace': 'con', 'h.c_pl050_max_std_perm_trace': 'con', 'h.c_pr001_sum_prem_trace ': 'con', 'h.c_pr002_prem_paid_trace': 'con', 'h.c_pr010_sum_prem_h_heal_trace': 'con', 'h.c_pr011_sum_prem_accid_trace': 'con', 'i.c_bs034_i_valid_policy730_count_trace': 'con', 'i.c_pl020_i_l_exp_count_trace': 'con', 'i.c_pl025_i_count_trace': 'con', 'i.c_pl216_i_h_count_trace': 'con', 'i.c_bs038_i_valid_policy180_count_trace': 'con', 'i.c_pl068_i_cpnst_count_lst5y_trace': 'con', 'i.c_pl194_i_num_claims_trace': 'con', 'i.c_pl091_total_claim_num_lst1y_trace': 'con', 'i.c_in072_i_insure_life_cii_times_trace': 'con', 'i.accu_risk_c_02': 'con', 'illness.total_insurance_hospitalized': 'con', 'illness.total_payment_hospitalized': 'con', 'illness.compensate_times_hospitalized': 'con', 'illness.total_insurance_outpatient': 'con', 'illness.total_payment_outpatient': 'con', 'illness.compensate_times_outpatient': 'con', 'x.s_bs020_seniority': 'con', 'x.s_bs021_sustainedrate': 'con', 'x.s_bs022_fyc': 'con', 'x.s_bs023_fycnum ': 'con', 'x.s_bs024_illegalpoint': 'con', '投保人身高': 'con', '投保人体重': 'con', '投保人BMI': 'con', '投保人吸烟年数': 'con', '投保人每天吸烟支数': 'con', '投保人吸烟指数': 'con', '投保人收入': 'con', '被保人身高': 'con', '被保人体重': 'con', '被保人BMI': 'con', '被保人吸烟年数': 'con', '被保人每天吸烟支数': 'con', '被保人吸烟指数': 'con', '被保人收入': 'con', 'risk_flag': 'target'}
    >>> config.get_columns_dict("na_value")
    {'投保人身高': '-1', '投保人体重': '-0.02', '投保人BMI': '-0.02', '投保人吸烟年数': '-1', '投保人每天吸烟支数': '-1', '投保人吸烟指数': '0.02', '投保人收入': '-0.02', '被保人身高': '-1', '被保人体重': '-1', '被保人BMI': '-0.02', '被保人吸烟年数': '-1', '被保人每天吸烟支数': '-1', '被保人吸烟指数': '0.02', '被保人收入': '-0.02'}
    >>> config.get_columns_dict("fill_na")
    {'投保人身高': 'mean'}
    >>> config.get_columns_list("ftype","con")
    ['投保人身高','投保人体重'...]
    """
    def __init__(self, config_file):
        self.config = read_xml(config_file)

    def get_attribute(self, attr: str):  # 用于获取xml中某个单独的字段下面的值
        """
        :param attr:
        :return:
        """
        new_dict = self.get_new_dict(attr)  # 获得指定节点构造的新字典new_dict
        if not new_dict.get(0):    # new_dict的第一个键值为空，则该指定节点下无子节点，
            return self.config.find(attr).text
        else:      # 指定节点下有子节点，遍历子节点并将节点值以列表形式返回
            field_value = []
            for child in self.config.iter(attr):
                for node in child:
                    field_value.append(node.text)
            return field_value

    def get_new_dict(self, attr: str):
        """
        将xml生成为dict,
        将指定节点添加到list中，将list转换为字典dict_init
        叠加生成多层字典dict_new，如：
        {0: {'name': 'e.rowkey', 'dtype': 'str', 'ftype': 'useless'},
         1: {'name': 'hn.ca_bxperiod', 'dtype': 'str', 'ftype': 'con'}}
        """
        dict_new = {}
        for key, value in enumerate(self.config.iter(attr)):
            dict_init = {}
            list_init = []
            for item in value:
                list_init.append([item.tag, item.text])
                for lists in list_init:
                    dict_init[lists[0]] = lists[1]
            dict_new[key] = dict_init
        return dict_new

    def get_columns_dict(self, dict_name: str):  # 以某个维度获得全部特征的对应属性
        keys = []
        values = []
        for name, info in self.get_new_dict("feature").items():  #遍历xml转换的字典
            if dict_name in info.keys():
                for key, value in info.items():
                    if key == 'name':
                        keys.append(value)
                    if key == dict_name:
                        values.append(value)
        return dict(zip(keys, values))      #返回以name值为键，对应属性为值的字典

    def get_columns_list(self, dict_name: str, value_name: str):
        """
        :param dict_name:
        :param value_name:
        :return:
        """
        columns_dict = self.get_columns_dict(dict_name)
        columns_list = [key for key, value in columns_dict.items() if value == value_name]
        return columns_list


class DataUtils:
    def __init__(self, config_file: str=None, config_parser: XmlConfigParser=None):
        if config_file is not None:
            self.config = XmlConfigParser(config_file)
        elif config_parser is not None:
            self.config = config_parser
        else:
            raise ValueError("this class needs one param at least")
        self.data = None
        self.bin_columns = None
        self.con_columns = None
        self.cat_columns = None
        self.target_column = None
        self.train_features = None

    def read_data(self, filename: str) -> pd.DataFrame:
        """
        :author: wangdaizheng
        :function: 通过配置文件中的配置读取数据
        :param filename: 数据文件
        :return: 处理过的读取数据
        """
        str_columns = {_: str for _ in self.config.get_columns_list("dtype", "str")}
        encoding = self.config.get_attribute("encode")
        separator = self.config.get_attribute("separator")
        na_values_dict_tmp = self.config.get_columns_dict("na_value")
        na_values_dict = {}
        for k, v in na_values_dict_tmp.items():
            if k in str_columns:
                na_values_dict[k] = v
            else:
                na_values_dict[k] = float(v)
        self.data = pd.read_csv(filename, encoding=encoding, na_values=na_values_dict, dtype=str_columns, sep=separator)
        return self.data

    def get_columns(self) -> [list, list, list, str]:                          # 封装获取列函数
        """
        :author: wangdaizheng
        :function: 从配置文件中读取不同种类的column list
        :return: bin_columns, con_columns, cat_columns, target_columns
        """
        self.bin_columns = self.config.get_columns_list("ftype", "bin")
        self.con_columns = self.config.get_columns_list("ftype", "con")
        self.cat_columns = self.config.get_columns_list("ftype", "cat")
        self.target_column = self.config.get_attribute("target_feature")
        return self.bin_columns, self.con_columns, self.cat_columns, self.target_column

    def filter_train_features(self)->list:
        """
        todo: 衍生变量怎么添加进来的问题
        :function:
        :return:
        """
        self.train_features = self.config.get_attribute("train_features")
        if self.bin_columns is not None:
            self.bin_columns = [_ for _ in self.bin_columns if _ in self.train_features]
        if self.con_columns is not None:
            self.con_columns = [_ for _ in self.con_columns if _ in self.train_features]
        if self.cat_columns is not None:
            self.cat_columns = [_ for _ in self.cat_columns if _ in self.train_features]
        return self.train_features


class LabelsEncoder:
    def __init__(self):
        self.labels_encoder = None
        self.columns = None

    def fit(self, data: pd.DataFrame):
        self.columns = data.columns
        self.labels_encoder = {i: LabelEncoder() for i in self.columns}
        [self.labels_encoder.get(i).fit(data[i]) for i in self.columns]

    def transform(self, data: pd.DataFrame, columns_method: str= "intersection"):
        fit_columns = self.columns
        transform_columns = data.columns
        if columns_method == "intersection":
            labels_encoded_list = [self.labels_encoder.get(col_name).transform(data[col_name]) for col_name in transform_columns if col_name in fit_columns]
            return np.array(labels_encoded_list).T
        elif columns_method == "raise":
            fit_columns_for_judge = [col for col in fit_columns if col in transform_columns]
            if len(fit_columns) == len(fit_columns_for_judge):
                labels_encoded_list = [encoder.transform(data[name]) for name, encoder in self.labels_encoder]
                return np.array(labels_encoded_list).T
            else:
                raise ValueError("columns length are not equel")
        else:
            print(columns_method+" method not completed, please try intersection or raise")
            return None

    def fit_transform(self, data: pd.DataFrame, columns_method: str= "intersection"):
        self.fit(data)
        return self.transform(data, columns_method)

    def inverse_transform(self, data: pd.DataFrame)->pd.DataFrame:
        inverse_columns = data.columns
        for i in inverse_columns:
            try:
                data[i] = self.labels_encoder.get(i).inverse_transform(data[i])
            except KeyError:
                data[i] = data[i]
        return data


class PreProcesser:
    def __init__(self):
        self.config = None
        self.labels_encoder = None

    def __call__(self, data_utils: DataUtils, one_hot: bool=True) -> DataUtils:
        self.config = data_utils.config
        self.labels_encoder = LabelsEncoder()
        #todo: 补值、label encoder、one hot encoder
        data = self.labels_encoder.fit_transform(data_utils.data[data_utils.cat_columns])
        if one_hot:
            self.one_hot.fit_transform()
        data_utils.data = data
        return data_utils


if __name__ == "__main__":
    from pprint import pprint
    # import doctest
    # doctest.testmod(verbose=True)  #doctest.testmod是测试模块，verbose默认是False,出错才用提示；True，对错都有执行结果
    test = XmlConfigParser("features.xml")
    print(test.get_attribute("test"))
    print(test.get_attribute("train_features"))
    # pprint(test.get_columns_dict("ftype"))
    data_utils = DataUtils("features.xml")
    pprint(data_utils.read_data("test_data_1000.csv"))
