# Mushroom Vision 4 🍄

This first project has the ambitious goal of building a deep learning model capable of classifying images of fungi into their species, providing information on the edibility or toxicity of each recognized species

The [Mushroom Vision 4 app](https://huggingface.co/spaces/simo98/MushroomVision4), currently running on HuggingFace, uses a DenseNet161 architecture to classify mushroom images into 4 classes.

The table below shows the species considered in this study (you can try these images by dragging and dropping them into the app).

|| **Species name** | **Common name** | **Photo** | **Edibility** |
|:----:|:----------------:|:---------------:|:-------------:|:-------------:|
|**01**| [Amanita Muscaria](https://wikipedia.org/wiki/Amanita_muscaria) | Cocco del monte | <img src="/Species/02_AmanitaMuscaria.jpg" alt="Cocco" width="250" height="150" /> | Toxic |
|**02**| [Amanita Vaginata](https://wikipedia.org/wiki/Amanita_vaginata) | Amanita| <img src="/Species/03_AmanitaVaginata.jpg" alt="Cocco" width="250" height="150" /> | Not edible |
|**03**| [Boletus Edulis](https://wikipedia.org/wiki/Boletus_edulis) | Porcino | <img src="/Species/07_BoletusEdulis.jpg" alt="Cocco" width="250" height="150" /> | Great |
|**04**| [Boletus Erythropus](https://it.wikipedia.org/wiki/Neoboletus_erythropus) | Cappella Malefica | <img src="/Species/08_BoletusErythropus.jpg" alt="Cocco" width="250" height="150" /> | Toxic when raw |

Feature extractors with 4 different architectures were tried:
1. AlexNet
2. DenseNet121
3. ResNet50
4. VGG16

The DenseNet121 performed better so all variants of this architecture available on torch were tested to identify the best performing:
1. DenseNet121
2. DenseNet161
3. DenseNet169
4. DenseNet201

The 161 form ouperformed the others as shown in the confusion matrices below

