#!/usr/bin/env python
# -*- coding:utf8 -*-
'''
处理文本文件char_dict,将该字典文件中unicode编码和HWDB1文件夹中的子目录对应上,同时对应到3755字的编码上。
'''
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

HWDB1_ROOT_DIR = '/mnt/sdb/ocr/HWDB1/train/'

char_to_idx = {}
idx_to_char = {}


def read_37555_file(path):
    '''
    将3755.txt解析成字典
    :param path:
    :return:
    '''
    with open(path, 'rt') as f:
        idx = 0
        for line in f:
            char = line.split('\r\n')[0]
            char_to_idx[char] = idx
            idx_to_char[idx] = char
            idx += 1
        print(char_to_idx)
        print(idx_to_char)


def read_deal_char_dict_file(path):
    with open(path, 'rt') as f:
        idx = 0
        char = ''
        char_dir = ''
        file1 = open("test1.txt", "w")
        for line in f:
            if line.startswith('V') or line.startswith('sV'):  # 字符
                char = line[line.index('\u'):len(line)-1].encode('utf-8').decode("unicode_escape")
                print(char)
                char_utf_8 = char.encode('utf-8')
                if char_utf_8 == '硷':
                    char = ''
                    continue
                # print(char,char_to_idx[char])

            if line.startswith('I'):  # 字符对应的手写汉字文件夹
                num = int(line.split('I')[1])
                char_dir = str("%05d" % num)
                if char != '' and char_dir != '':  # 一个汉字对应打目录
                    # file1.write(char)
                    char_real_dir =  os.path.join(HWDB1_ROOT_DIR,char_dir)
                    for parent, dirnames, filenames in os.walk(char_real_dir, followlinks=True):
                        for filename in filenames:
                            if int(char_to_idx[char_utf_8]) < 10:
                                file_path = os.path.join(parent, filename)
                                file1.write(str(char_to_idx[char_utf_8]))
                                file1.write(" ")
                                file1.write(file_path)
                                file1.write(" ")
                                file1.write(char_utf_8)
                                file1.write(" ")
                                file1.write("\r\n")
                    # print("%s,%s,%s,%s",char, char_utf_8, char_to_idx[char_utf_8], char_dir)
                    char = ''
                    char_dir = ''
        file1.close()

def main():
    file_3755_path = '3755.txt'
    read_37555_file(file_3755_path)

    char_dict_file = 'char_dict'
    read_deal_char_dict_file(char_dict_file)

if __name__ == '__main__':
    main()