from jieba import analyse
import jieba_word_segmentation.skateboard_lyric as skateboard_lyric

print('tf-idf method:')
tags=analyse.extract_tags(skateboard_lyric.lyric)
for tag in tags:
    print(tag,end='/')

print('\n\ntextrank method:')
tags=analyse.textrank(skateboard_lyric.lyric)
for tag in tags:
    print(tag,end='/')

"""
tf-idf method:
摩擦/滑板鞋/一步/光滑/地上/我要/魔鬼/两步/步伐/天黑/月光/时尚/美丽/华晨/不怕/庞麦郎/有时/真的/约瑟/自己/

textrank method:
时尚/时间/滑板鞋/摩擦/看到/寻找/美丽/城市/时刻/月光/完成/驱使/夜幕/力量/街道/魔鬼/步伐/喜欢/脚步/舞蹈/
"""