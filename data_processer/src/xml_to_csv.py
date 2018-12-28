#!/usr/bin/python
# -*- coding: utf-8 -*-

from src.data_xml import XmlConfigParser
import pandas as pd


class XmlToCsv(XmlConfigParser):
    def get_csv(self, file_name: str, encoding: str):
        """
        将获得的新字典dict_new转化为DataFrame,然后转化为CSV输出
        """
        node_list = []
        for name, info in self.get_new_dict("feature").items():  #遍历xml转换的字典
            node_list.append(info)

        df = pd.DataFrame(node_list)
        cols = list(df.columns)
        cols.insert(0, cols.pop(cols.index('name')))  #将列索引name插到首列位置
        df = df.ix[:, cols]
        df.to_csv(file_name, encoding=encoding)


if __name__ == "__main__":
    test = XmlToCsv("features.xml")
    test.get_csv("Xmltocsv.csv", encoding="GB2312")
