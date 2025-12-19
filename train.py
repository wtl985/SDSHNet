# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
from SDSHNet import RTDETR


if __name__ == '__main__':

    model = RTDETR(model=r'/SDSHNet/cfg/models/SDSHNet.yaml')
    #model.load('SDSHNet.pt')
    model.train(data=r'/SDSHNet/jiazawu.yaml',
                imgsz=640,
                epochs=1,
                batch=4,
                project='runs/train',
                name='exp',
                workers=0,
                amp=False
                )
