"""
程序运行中会用到的异常类在此定义
"""


class IllegalArgumentException(Exception):
    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message
