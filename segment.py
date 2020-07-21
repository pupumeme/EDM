# -*- coding: utf-8 -*-

import jieba
import json 
import re
def main():

    jieba.set_dictionary('jieba_dict/dict.txt.big')

    # load stopwords set
    stopword_set = set()
    with open('jieba_dict/stopwords.txt','r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))

    output = open('wiki_seg_to_doc.txt', 'w', encoding='utf-8')
    with open('wiki_data.json', 'r', encoding='utf-8') as content :
    # with open('wiki_data.json', 'r', encoding='utf-8') as content :
        wiki_json=json.load(content)
        i=1
        for key, value in wiki_json.items():
            # value = re.split('。|！|？|\n',value)
            # print(value)
            # for line in value:
            words = jieba.cut(value, cut_all=False)
            # clean words whos with only numbers and characters
            words = [w for w in words if not re.match('^[a-z|A-Z|0-9|.|\ ]*$',w) \
                                        and w not in stopword_set]
            if len(words) >1 :
                # print(words)
                for word in words:
                    output.write(word + ' ')
                # output.write('\n')
                output.write('\n')
            if i % 10000 == 0:
                print("已完成前 %d 行的斷詞" % (i))
            i+=1
    output.close()

if __name__ == '__main__':
    main()
