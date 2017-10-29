# encoding=utf-8
import jieba
import jieba.posseg as pseg

import jieba_word_segmentation.skateboard_lyric as skateboard_lyric


jieba.add_word('约瑟翰')
jieba.add_word('庞麦郎')
jieba.add_word('华晨宇')


lyric_arr=skateboard_lyric.lyric.split('\n')

for sentence in lyric_arr:
    for (word, flag) in pseg.cut(sentence):
        print(word,flag,end='/ ')
    print('')

"""
我 r/ 的 uj/ 滑板鞋 n/ 时尚 n/ 时尚 n/ 最 d/ 时尚 n/ 
回家 n/ 的 uj/ 路上 s/ 我 r/ 情不自禁 i/ 
摩擦 vn/   x/ 摩擦 vn/   x/ 摩 nz/ 
在 p/ 这 r/ 光滑 a/ 的 uj/ 地上 s/ 摩擦 vn/ 
摩擦 vn/ 
"""

for sentence in lyric_arr:
    print(jieba.lcut(sentence))
"""
['有些', '事', '我', '都', '已', '忘记']
['但', '我', '现在', '还', '记得']
['在', '一个', '晚上', '我', '的', '母亲', '问', '我']
['今天', '怎么', '不', '开心']
"""