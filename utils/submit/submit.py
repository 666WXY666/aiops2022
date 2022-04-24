'''
Copyright: Copyright (c) 2021 WangXingyu All Rights Reserved.
Description: 
Version: 
Author: WangXingyu
Date: 2022-04-20 10:31:37
LastEditors: WangXingyu
LastEditTime: 2022-04-24 22:26:03
'''
"""
Example for submit.

Before running this script, execute

    pip3 install requests
"""


# 提交答案服务域名或IP, 将在赛前告知
import json
import requests
HOST = "http://10.3.2.40:30083"
# 团队标识, 可通过界面下方权限获取, 每个ticket仅在当前赛季有效，如未注明团队标识，结果不计入成绩
TICKET = "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiIxNTAyMTQ3MjI3NjU2MDA3NzA5IiwiaWF0IjoxNjUwODEwMzQ5LCJ0aWNrZXQiOnsidGlkIjoiMTUwMjE0NzIyNzY1NjAwNzcwOSIsImNpZCI6IjE0OTYzOTg1MjY0Mjk3MjQ3NjAiLCJzZWFzb24iOiIxIiwic3RhcnQiOiIxNjUwMzg0MDAwMDAwIiwiZW5kIjoiMTY1MjYzMDM5OTAwMCJ9LCJpc3MiOiJCaXpzZWVyIiwiZXhwIjoxNjUyNjMwMzk5fQ.HH48VoYYlZxvSjsvcGXsjgROkckdJwLko5Dwc_yA1IwCDgOVOn_wZ_1-7Y8ExTQ6fjcokggHnWkrqKVGPFi44Q"


def submit(ctx):
    assert (isinstance(ctx, list))
    assert (len(ctx) == 2)
    assert (isinstance(ctx[0], str))
    assert (isinstance(ctx[1], str))
    data = {'content': json.dumps(ctx, ensure_ascii=False)}
    r = requests.post(
        url='%s/answer/submit' % HOST,
        json=data,
        headers={"ticket": TICKET}
    )
    return r.text


if __name__ == '__main__':
    '''
        test part
    '''
    res = submit(["adservice", "k8s容器CPU压力"])
    print(res)
    # {"code":0,"msg":"","data":1} 成功提交一次答案
