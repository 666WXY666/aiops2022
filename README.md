<!--
 * @Copyright: Copyright (c) 2022 WangXingyu All Rights Reserved.
 * @Description: 
 * @Version: 
 * @Author: WangXingyu
 * @Date: 2022-04-08 11:42:26
 * @LastEditors: WangXingyu
 * @LastEditTime: 2022-05-10 22:39:18
-->
# AIOPS2022——Aurora_BUPT团队解决方案

比赛详情：https://666wxy666.notion.site/AIOPS2022-fe5757f636574b5c8c49377127e9fe49

#### 本方案采用SPOT异常检测（离线训练模型，在线检测异常）+改进的macroRCA+PageRank进行根因定位+catboost故障分类

## 项目结构

- algorithm文件夹主要是算法相关代码
- data文件夹是相关数据
- example文件夹是数据分析可视化样例代码
- model文件夹是训练的离线模型
- result文件夹是结果
- utils文件夹是工具类代码，包括数据获取、数据预处理、数据可视化、结果提交等代码
- main.py为主函数和整个程序入口
- requirements.txt为所需的环境（python环境为anaconda的python3.8）

## 离线训练方法

在main.py中将type设为'train'，在data中放入训练数据，执行`python main.py`即可进行离线训练，离线训练的模型将会放入model文件夹内。

## 离线测试方法

在main.py中将type设为'offline_test'，在data中放入离线测试数据，执行`python main.py`即可进行离线测试，离线测试的结果将会显示在屏幕上。

## 在线测试方法

在main.py中将type设为'online_test'，执行`nohup python main.py >run.log 2>&1 &`即可进行后台的在线测试，在线测试会读取在线流式数据，并且将结果提交。
