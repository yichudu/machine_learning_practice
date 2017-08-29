# coding=utf-8
import os


def create_file(absolute_path):
    """
    调用方式 形如 create_file('d:/yc/yi/a.txt')
    :param absolute_path: 
    :return: 
    """
    last_slash_index = absolute_path[::-1].index('/') * -1
    dir_path = absolute_path[:last_slash_index]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file = open(absolute_path, 'w')
    file.close()
