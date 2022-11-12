import layoutparser as lp
import pdf2image
import numpy as np
from gibberish_detector import detector
# gibberish detector from  https://pypi.org/project/gibberish-detector/

class LayoutParserManager:
    def __init__(self):
        self.model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
        self.layout = None
        self.ocr_agent = lp.TesseractAgent(languages='eng')
        self.gibberish_detector = detector.create_from_model('model/big.model')

    def getModel(self):
        return self.model

    def getImage(self, input_pdf):
        return np.asarray(pdf2image.convert_from_path(input_pdf)[0])

    def drawResult(self, img, layout):
        res = lp.draw_box(img, layout, box_width=3)
        return res

    def onlyTextBlocks(self, layout_result):
        #lp.Layout([b for b in layout_result if b.type == 'Text'])
        # return all text boxes
        return lp.Layout([b for b in layout_result])

    def getTextsInTextBlock(self, text_blocks, img):
        for block in text_blocks:
            # Crop image around the detected layout
            segment_image = (block
                             .pad(left=15, right=15, top=5, bottom=5)
                             .crop_image(img))

            # Perform OCR
            text = self.ocr_agent.detect(segment_image)
            if self.gibberish_detector.is_gibberish(text):
                print(text)
                text = 'gibberish'
            # Save OCR result
            block.set(text=text, inplace=True)
