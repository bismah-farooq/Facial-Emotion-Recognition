| Model | Params | Best epoch | Test acc | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|---|
| Custom CNN | 4.82M | 56 | 0.6414 | 0.5944 | 0.6372 |
| Custom CNN [cnn_noweights] | 4.82M | 59 | 0.6704 | 0.6051 | 0.6639 |
| Custom CNN [cnn_sqrtw] | 4.82M | 66 | 0.6817 | 0.6531 | 0.6787 |
| ResNet-18 (fine-tuned) | 11.17M | 20 | 0.6609 | 0.6116 | 0.6615 |

| Class | Custom CNN | Custom CNN [cnn_noweights] | Custom CNN [cnn_sqrtw] | ResNet-18 (fine-tuned) |
|---|---|---|---|---|
| angry | 0.583 | 0.607 | 0.622 | 0.592 |
| disgust | 0.427 | 0.327 | 0.589 | 0.386 |
| fear | 0.400 | 0.463 | 0.485 | 0.486 |
| happy | 0.862 | 0.876 | 0.886 | 0.869 |
| neutral | 0.618 | 0.645 | 0.646 | 0.633 |
| sad | 0.508 | 0.550 | 0.551 | 0.543 |
| surprise | 0.763 | 0.769 | 0.792 | 0.772 |

Top confusions (true -> predicted):
- Custom CNN: sad->neutral (274), fear->sad (197), neutral->sad (157), fear->surprise (155)
- Custom CNN [cnn_noweights]: sad->neutral (250), fear->sad (223), neutral->sad (175), fear->angry (133)
- Custom CNN [cnn_sqrtw]: sad->neutral (239), fear->sad (190), sad->fear (145), sad->angry (144)
- ResNet-18 (fine-tuned): sad->neutral (231), fear->sad (178), neutral->sad (165), sad->fear (130)
