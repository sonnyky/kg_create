from LayoutManager.LayoutParserManager import LayoutParserManager as LPM
# Penn reebank https://hanlp.hankcs.com/docs/annotations/constituency/ptb.html

# POS_TAGS : https://hanlp.hankcs.com/docs/annotations/constituency/ptb.html
# pos tags: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

lp = LPM()

with open('extracted_text.txt', 'r', encoding='utf-8') as f:
    lines = [line.rstrip() for line in f.readlines() if line.strip()]

NP_objects = lp.gatherNP(lp.constituencyParsing(lines[0]))
with open('cp.txt', 'w', encoding='utf-8') as f:
    f.write(str(lp.constituencyParsing(lines[0])))
