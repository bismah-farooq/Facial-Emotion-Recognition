| Model | Params | Best epoch | Test acc | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|---|
| Custom CNN | 4.82M | 56 | 0.6414 | 0.5944 | 0.6372 |
| ResNet-18 (fine-tuned) | 11.17M | 20 | 0.6609 | 0.6116 | 0.6615 |

| Class | Custom CNN | ResNet-18 (fine-tuned) |
|---|---|---|
| angry | 0.583 | 0.592 |
| disgust | 0.427 | 0.386 |
| fear | 0.400 | 0.486 |
| happy | 0.862 | 0.869 |
| neutral | 0.618 | 0.633 |
| sad | 0.508 | 0.543 |
| surprise | 0.763 | 0.772 |

Top confusions (true -> predicted):
- Custom CNN: sad->neutral (274), fear->sad (197), neutral->sad (157), fear->surprise (155)
- ResNet-18 (fine-tuned): sad->neutral (231), fear->sad (178), neutral->sad (165), sad->fear (130)
