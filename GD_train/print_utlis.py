#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import time

# 支持在python2、3下调用本文件


class Log(object):

# 用例
# 从log = Log('test_data')开始，print会向文件'test_data'与屏幕输出（双向输出）
# 包括：
#   - 凡print函数皆双向输出
#     - 此处代码中的print
#     - 被调用的函数中有print
#     - 被import的文件中有print
#   - 支持输出到文件、屏幕的flush()
#   - 不支持'\r'输出到文件（vim下'\r'会显示为^M），支持'\r'输出到文件，故进度条只能在屏幕上显示
# 直到log.close()，print才变为只输出到屏幕

    def __init__(self, filename='',mode='w',*args):
        # filename =
         # time.strftime("%m-%d_%H:%M:%S", time.localtime())+filename
        self.f = open(filename, mode)
        sys.stdout = self
        print('====== log start ======',
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            # 格式化成2016-03-20 11:45:39形式
            '======')

    def write(self, data):
        self.f.write(data)
        sys.__stdout__.write(data)

    def flush(self):
        # 例
        #   print("xxxx", end='')
        #   log.flush()
        # 可使"xxxx"立即输出到屏幕和文件；
        # 若不"log.flush()"，则要等到之后有换行的 print("xxxx")
        self.f.flush()
        sys.__stdout__.flush()

    def close(self):
        print('======= log end =======',
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            '======')
        self.flush()
        self.f.close()
        sys.stdout=sys.__stdout__



def cursor_back(func):
    # 光标回到所在行首，且只输出到屏幕，不输出到别处
        # 因为在屏幕是上，'\r'可令光标返回行首，但本行已输出的字符不会删除
        # 但在文件中，'\r'显示为"^M" 或 "<0x0d>"，不能返回光标到行首
    def func_with_cursor_back(*args, **kw):
        # 以防之前stdout修改，如"屏幕、文件双向输出"功能
        original = sys.stdout
        sys.stdout = sys.__stdout__
        # 光标回到所在行首
        result = func(*args, **kw)
        print("\r",end="")
        sys.stdout.flush()
        # 还原sys.stdout
        sys.stdout.flush()
        sys.stdout = original

        return result
    return func_with_cursor_back



def progess_bar(process_rate, precision=2,num_block=30,prefix="",suffix="", clean=True):
    # 输出到屏幕
        # process_rate 是[0,1]的float，表示百分之多少的进度
        # precision小数点位数
        # 进度条的'>'块数
    # 效果
        #  94.000 %   | > > > > > > > > > > > > > > > > > >     |
    # clean:    当process_rate == 1时怎么结束
        # Ture:  清除进度条，不换行，光标回行头
        # False: 保留进度条，直接换行

    line=""
    line+=prefix
    # 右对齐输出百分数
    line+= ( ('%%%ds'%(precision+4))  % (('%%.%df'%precision)%(100*process_rate)) ) +' %'
    line+='   |'
    # 输出进度条
    n = int(process_rate*num_block)
    for i in range(n):
        line+='>'
    for i in range(num_block-n):
        line+=' '
    line+='|   '
    # 后缀
    line+=suffix

    cursor_back(print)(line, end='')

    if process_rate==1:
        if clean:
            cursor_back(print)(' '*len(line), end='')
        else:
            print()

