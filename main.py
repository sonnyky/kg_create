from LayoutManager.LayoutParserManager import LayoutParserManager as LPM
import pdf2image
import cv2 as cv
import numpy as np

# convert pdf to image


# First step, we break down to titles and headings of the document
lp_manager = LPM()
pdf_input = './documents/allianz_basic_9-9.pdf'
image_input = pdf2image.convert_from_path(pdf_input)
img = np.array(image_input[0])

model = lp_manager.getModel()
layout = model.detect(img)

result = lp_manager.drawResult(img, layout)
cv.imwrite('./result_temp.jpg', np.array(result))
cv.imshow('', np.array(result))
cv.waitKey(0)
