import layoutparser as lp
import pdf2image
import numpy as np

class LayoutParserManager:
    def __init__(self):
        self.model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
        self.layout = None

    def getModel(self):
        return self.model

    def getImage(self, input_pdf):
        return np.asarray(pdf2image.convert_from_path(input_pdf)[0])

    def drawResult(self, img, layout):
        res = lp.draw_box(img, layout, box_width=3)
        return res