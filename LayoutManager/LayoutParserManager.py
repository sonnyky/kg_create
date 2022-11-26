import layoutparser as lp
import pdf2image
import numpy as np
from gibberish_detector import detector
# gibberish detector from  https://pypi.org/project/gibberish-detector/
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import json

# check LiL

class LayoutParserManager:
    def __init__(self):
        self.model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
        self.layout = None
        self.ocr_agent = lp.TesseractAgent(languages='eng')
        self.gibberish_detector = detector.create_from_model('model/big.model')
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")

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

    def constituencyParsing(self, sentence):
        return self.predictor.predict(sentence)

    # the Json is as returned by constituencyParsing method
    def gatherNP(self, sentence_json):
        cur_root = sentence_json["hierplane_tree"]["root"]

        if cur_root["nodeType"] != "S":
            pass
        else:
            #here we loop all the nodes of the sentence tree and assign them to a graph node/relationship
            # check the length of children words
            self.checkThisNode(cur_root)
        print(cur_root)

    def checkThisNode(self, node):
        word = node["word"]
        nodeType = node["nodeType"]
        if nodeType == "NP":
            print(word + " is NP")
        if "children" in node.keys() and len(node["children"]) > 0:
            for child in node["children"]:
                self.checkThisNode(child)
