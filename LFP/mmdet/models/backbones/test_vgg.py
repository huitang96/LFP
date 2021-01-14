'''
# import os
# import time
import datetime

print('Hello World!')
print('Time is ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %A'))
print('__name__ value: ', __name__)


def main():
    print('this message is from main function')


if __name__ == '__main__':
    main()
    # print(__name__)
'''

import torch
import torch.nn as nn
#from .ssd_vgg import SSDVGG
import ssd_vgg.SSDVGG
print('Hello World!')



def main():
    print('this message is from main function')


if __name__ == '__main__':
    main()
    # print(__name__)