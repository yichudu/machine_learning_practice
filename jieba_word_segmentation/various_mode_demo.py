# encoding=utf-8
import jieba
import jieba.posseg as pseg

text='吊带背心'+'时尚秋装'+'红色连衣裙'
seg_list = jieba.cut(text, cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut(text, cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut(text)  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search(text)  # 搜索引擎模式
print(", ".join(seg_list))

"""
Full Mode: 吊带/ 背心/ 时尚/ 秋装/ 红色/ 连衣/ 连衣裙/ 衣裙
Default Mode: 吊带/ 背心/ 时尚/ 秋装/ 红色/ 连衣裙
吊带, 背心, 时尚, 秋装, 红色, 连衣裙
吊带, 背心, 时尚, 秋装, 红色, 连衣, 衣裙, 连衣裙
"""

for (word, flag) in pseg.cut(text):
    print(word,flag,end='/ ')
print('\n')
"""
吊带 n/ 背心 n/ 时尚 n/ 秋装 n/ 红色 n/ 连衣裙 nr/ 
"""


for (word, flag) in pseg.cut('我的心是个落叶的季节'):
    print(word,flag,end='/ ')