"""
Example for submit.

Before running this script, execute

    pip3 install requests
"""
import json

import requests

# 提交答案服务域名或IP, 将在赛前告知
HOST = "http://10.3.2.40:30083"
# 团队标识, 可通过界面下方权限获取, 每个ticket仅在当前赛季有效，如未注明团队标识，结果不计入成绩
TICKET = "your team ticket"


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
