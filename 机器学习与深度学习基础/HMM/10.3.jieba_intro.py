# !/usr/bin/python
# -*- coding:utf-8 -*-

import jieba
import jieba.posseg


if __name__ == "__main__":
    f = open('novel.txt', encoding='utf-8')
    str = f.read()
    f.close()

    seg = jieba.posseg.cut(str)
    for s in seg:
        print(s.word, s.flag, '|', end=' ')
        # print(s.word, '|', end = '')

    print('='*60)
    words = jieba.cut(str)
    for word in words:
        print(word, end='|')