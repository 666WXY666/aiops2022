'''
Copyright: Copyright (c) 2021 WangXingyu All Rights Reserved.
Description: 
Version: 
Author: WangXingyu
Date: 2022-04-20 10:31:37
LastEditors: WangXingyu
LastEditTime: 2022-04-20 10:31:39
'''
"""
Example for data consuming.

Before running this script, execute

    pip3 install kafka-python
"""


import json
from kafka import KafkaConsumer
AVAILABLE_TOPICS = {
    'kpi-1c9e9efe6847bc4723abd3640527cbe9',
    'metric-1c9e9efe6847bc4723abd3640527cbe9',
    'trace-1c9e9efe6847bc4723abd3640527cbe9',
    'log-1c9e9efe6847bc4723abd3640527cbe9'
}

CONSUMER = KafkaConsumer(
    'kpi-1c9e9efe6847bc4723abd3640527cbe9',
    'metric-1c9e9efe6847bc4723abd3640527cbe9',
    'trace-1c9e9efe6847bc4723abd3640527cbe9',
    'log-1c9e9efe6847bc4723abd3640527cbe9',
    bootstrap_servers=['10.3.2.41', '10.3.2.4', '10.3.2.36'],
    auto_offset_reset='latest',
    enable_auto_commit=False,
    security_protocol='PLAINTEXT'
)


def main():
    """Consume data and react"""
    assert AVAILABLE_TOPICS <= CONSUMER.topics(), 'Please contact admin'
    print('test consumer')
    i = 0
    for message in CONSUMER:
        i += 1
        data = json.loads(message.value.decode('utf8'))
        print(type(data), data)


if __name__ == '__main__':
    '''
        start to consume kafka
    '''
    main()
