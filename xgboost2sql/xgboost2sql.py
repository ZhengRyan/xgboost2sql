#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: xgboost2sql.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-05-17
'''

import codecs
import json
import math
import warnings

import xgboost


class XGBoost2Sql:
    sql_str = '''
    select {0},1 / (1 + exp(-(({1})+({2})))) as score
    from (
    select {0},
    {3}
    from {4})
    '''
    code_str = ''

    def transform(self, xgboost_model, keep_columns=['key'], table_name='data_table'):
        """

        Args:
            xgboost_model:训练好的xgboost模型
            keep_columns:使用sql语句进行预测时，需要保留的列。默认主键保留
            table_name:待预测数据集的表名

        Returns:xgboost模型的sql预测语句

        """

        strs = self.get_dump_model(xgboost_model)
        ### 读模型保存为txt的文件
        # with open('xgboost_model', 'r') as f:
        #     strs = f.read()
        ### 读模型保存为txt的文件

        logit = self.get_model_config(xgboost_model)

        # 解析的逻辑
        columns_l = []
        tree_list = strs.split('booster')
        for i in range(1, len(tree_list)):
            tree_str = tree_list[i]
            lines = tree_str.split('\n')
            v_lines = lines[1:-1]

            self.code_str += '--tree' + str(i) + '\n'
            is_right = False
            self.pre_tree(v_lines, is_right, 1)
            columns_l.append('tree_{}_score'.format(i))
            if i == len(tree_list) - 1:
                self.code_str += '\t\tas tree_{}_score'.format(i)
            else:
                self.code_str += '\t\tas tree_{}_score,\n'.format(i)
        columns = ' + '.join(columns_l)
        self.sql_str = self.sql_str.format(','.join(keep_columns), columns, logit, self.code_str, table_name)
        return self.sql_str

    def get_dump_model(self, xgb_model):
        """

        Args:
            xgb_model:xgboost模型

        Returns:

        """
        if isinstance(xgb_model, xgboost.XGBClassifier):
            xgb_model = xgb_model.get_booster()
        # joblib.dump(xgb_model, 'xgb.ml')
        # xgb_model.dump_model('xgb.txt')
        # xgb_model.save_model('xgb.json')
        ret = xgb_model.get_dump()
        tree_strs = ''
        for i, _ in enumerate(ret):
            tree_strs += 'booster[{}]:\n'.format(i)
            tree_strs += ret[i]
        return tree_strs

    def get_model_config(self, xgb_model):
        """

        Args:
            xgb_model:xgboost模型

        Returns:

        """
        if isinstance(xgb_model, xgboost.XGBClassifier):
            xgb_model = xgb_model.get_booster()

        try:
            ###-math.log((1 / x) - 1)
            x = float(json.loads(xgb_model.save_config())['learner']['learner_model_param']['base_score'])
            return -math.log((1 - x) / x)
        except:
            warnings.warn(
                'xgboost model version less than :: 1.0.0, If the base_score parameter is not 0.5 when developing the model, Insert the base_score value into the formula "-math.log((1-x)/x)" and replace the -0.0 value at +(-0.0) in the first sentence of the generated sql statement with the calculated value')
            warnings.warn(
                'xgboost 模型的版本低于1.0.0，如果开发模型时， base_score 参数不是0.5，请将base_score的参数取值带入"-math.log((1 - x) / x)"公式，计算出的值，替换掉生成的sql语句第1句中的+(-0.0)处的-0.0取值')
            return -0.0

    def pre_tree(self, lines, is_right, n):
        """

        Args:
            lines:二叉树行
            is_right:是否右边
            n:第几层

        Returns:

        """
        n += 1
        res = ''
        if len(lines) <= 1:
            str = lines[0].strip()
            if 'leaf=' in str:
                tmp = str.split('leaf=')
                if len(tmp) > 1:
                    if is_right:
                        format = '\t' * (n - 1)
                        res = format + 'else\n' + format + '\t' + tmp[1].strip() + '\n' + format + 'end'
                    else:
                        format = '\t' * n
                        res = format + tmp[1].strip()
            self.code_str += res + '\n'
            return
        v = lines[0].strip()
        start_index = v.find('[')
        median_index = v.find('<')
        end_index = v.find(']')
        v_name = v[start_index + 1:median_index].strip()
        v_value = v[median_index:end_index]
        ynm = v[end_index + 1:].strip().split(',')
        yes_v = int(ynm[0].replace('yes=', '').strip())
        no_v = int(ynm[1].replace('no=', '').strip())
        miss_v = int(ynm[2].replace('missing=', '').strip())
        z_lines = lines[1:]

        if is_right:
            format = '\t' * (n - 1)
            res = res + format + 'else' + '\n'
        if miss_v == yes_v:
            format = '\t' * n
            res = res + format + 'case when (' + v_name + v_value + ' or ' + v_name + ' is null' + ') then'
        else:
            format = '\t' * n
            res = res + format + 'case when (' + v_name + v_value + ' and ' + v_name + ' is null' + ') then'
        self.code_str += res + '\n'
        left_right = self.get_tree_str(z_lines, yes_v, no_v)

        left_lines = left_right[0]
        right_lines = left_right[1]
        self.pre_tree(left_lines, False, n)
        self.pre_tree(right_lines, True, n)
        if is_right:
            format = '\t' * (n - 1)
            self.code_str += format + 'end\n'

    def get_tree_str(self, lines, yes_flag, no_flag):
        """

        Args:
            lines:二叉树行
            yes_flag:左边
            no_flag:右边

        Returns:

        """
        res = []
        left_n = 0
        right_n = 0
        for i in range(len(lines)):
            tmp = lines[i].strip()
            f_index = tmp.find(':')
            next_flag = int(tmp[:f_index])
            if next_flag == yes_flag:
                left_n = i
            if next_flag == no_flag:
                right_n = i
        if right_n > left_n:
            res.append(list(lines[left_n:right_n]))
            res.append(list(lines[right_n:]))
        else:
            res.append(lines[left_n:])
            res.append(lines[right_n:left_n])
        return res

    def save(self, filename='xgb_model.sql'):
        """

        Args:
            filename:sql语句保存的位置

        Returns:

        """
        with codecs.open(filename, 'w', encoding='utf-8') as f:
            f.write(self.sql_str)


if __name__ == '__main__':
    ###训练1个xgboost二分类模型
    import xgboost as xgb
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=10000,
                               n_features=10,
                               n_informative=3,
                               n_redundant=2,
                               n_repeated=0,
                               n_classes=2,
                               weights=[0.7, 0.3],
                               flip_y=0.1,
                               random_state=1024)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1024)

    ###训练模型
    model = xgb.XGBClassifier(n_estimators=3)
    model.fit(X_train, y_train)
    xgb.to_graphviz(model)

    ###使用xgboost2sql包将模型转换成的sql语句
    xgb2sql = XGBoost2Sql()
    sql_str = xgb2sql.transform(model)
    print(sql_str)
    xgb2sql.save()
