"""
程序中会用到的异常类
"""


class IllegalArgumentException(Exception):
    def __init__(self, msg=""):
        self.msg = msg

    def __str__(self):
        return self.msg
