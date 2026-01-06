# SDSHNet
SDSHNet:Dynamic Feature Fusion with Transformer and Star Operation for Efficient Detection in Aluminum Alloys Microscopic Inclusion

This work presents a lightweight detection framework that combines a Transformer encoder with StarNet as the main feature extractor. StarNet relies on star shaped sparse attention and depth wise large kernel convolutions to reduce parameter count and computation load while maintaining expressive features. A dynamic feature fusion module is designed to adaptively enhance feature representation in key regions, significantly improving the detection capability for tiny, low-contrast inclusions.

Training progression of mAP@0.5 and mAP@[0.5:0.95] metrics for various detection approaches.
![fig6](https://github.com/user-attachments/assets/0702db7f-56be-4e95-9746-c74caa7098c6)



Comparison of visualization results for six types of aluminum alloy inclusions using different backbone networks including VGG-16, Darknet-53, GELAN, CSPNet, LSKNet, and R-ELAN.
![fig7](https://github.com/user-attachments/assets/35ffbb48-6972-4a62-9768-9ac423f52a43) 



Comparison of visualization results for six types of aluminum alloy inclusions using different backbone networks including ResNet-18, HGNetv2-1, HGNetv2-x, ResNet-50, ResNet-101, and SDSHNet.
![fig8](https://github.com/user-attachments/assets/f3fa041c-073f-4e72-81cb-535042c393d2)





The correlation between inference time and mAP alongside FLOPs metrics.
![fig11](https://github.com/user-attachments/assets/f1e64764-e87e-431b-9624-5d14d17c27f6)




Ablation Analysis of Model Components During Training
![fig12](https://github.com/user-attachments/assets/7fb824a3-64fa-4a1a-b6b9-f8e5cff07d75)
