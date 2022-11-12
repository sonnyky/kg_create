from LayoutManager.LayoutParserManager import LayoutParserManager as LPM
import pdf2image
import cv2 as cv
import numpy as np

# reference: https://towardsdatascience.com/analyzing-document-layout-with-layoutparser-ed24d85f1d44
# model zoo: https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html#model-label-map
# gibberish detector: https://pypi.org/project/gibberish-detector/

# First step, we break down to titles and headings of the document
lp_manager = LPM()

# convert pdf to image
pdf_input = './documents/allianz_basic_9-9.pdf'
image_input = pdf2image.convert_from_path(pdf_input)
img = np.array(image_input[0])

model = lp_manager.getModel()
layout = model.detect(img)

result = lp_manager.drawResult(img, layout)
cv.imwrite('./result.jpg', np.array(result))
cv.imshow('', np.array(result))
cv.waitKey(0)
text_blocks = lp_manager.onlyTextBlocks(layout)
# second step, get all texts in the layout
texts = lp_manager.getTextsInTextBlock(text_blocks, img)
with open('extracted_text.txt', 'w', encoding='utf-8') as f:
    for txt in text_blocks.get_texts():
        f.write(txt)
        f.write('\n')

