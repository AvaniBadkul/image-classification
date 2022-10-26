# Image Classification Using RESNET18

<!--
*** Thanks for checking out this README Template. If you have a suggestion that would
*** make this better, please fork the repo and create a pull request or simply open
*** an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->





<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for build-url, contributors-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors](https://img.shields.io/badge/Contributors-1-<green>.svg)](https://github.com/AvaniBadkul/image-classification/graphs/contributors)
[![MIT License][license-shield]][license-url]
<!--[![LinkedIn][linkedin-shield]][linkedin-url]-->



<!-- PROJECT LOGO -->



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [The Libraries](#the-libraries)
* [Usage](#usage)
* [Dataset](#dataset)
* [The Model](#the-model)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)


# Image Classification Using RESNET18

## About the Project
This repository contains my attempt at image classification of Buildings, Forests, and Mountains. The model used is 'RESNET18'.
The total number of images used are 600 (200 of each category). An Accuracy of 98.334% was achieved in under 5 epochs, each epoch
roughly took 2 minutes, hence the model was succcesfully trained in under 10 minutes. The images were augmented before being given
as an input to the RESNET18 model, they were resized to the dimensions of 224x224. The training-validation split was taken as 0.2.



## The Libraries

- fastai
- PIL
- ipywidgets

## Usage
- Clone the repo to download the 'export.pkl' file which contains the model.
- Open the .ipynb file in Jupyter Notebook or Google Colab.
- Make sure the 'export.pkl' file exists
- Run the following cells

```python
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```

```python
from fastbook import *
from fastai.vision.widgets import *
```

```python
path = Path()
path.ls(file_exts='.pkl')
learn_inf = load_learner(path/'export.pkl')
```

```python
btn_run = widgets.Button(description='Classify')
btn_upload = widgets.FileUpload()
```

```python
def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
```

```python
VBox([widgets.Label('Select your Image!'),
      btn_upload, btn_run, out_pl, lbl_pred])
```
- A widget will appear which is shown below and you can upload and classify the image.
- ![image](https://user-images.githubusercontent.com/79955028/113575593-b9d65380-963b-11eb-9eeb-c2516571d8f2.png)




## Dataset  
The Dataset taken to generate the model was taken from [Kaggle](https://www.kaggle.com/puneet6060/intel-image-classification)
![image](https://user-images.githubusercontent.com/79955028/113575232-3288e000-963b-11eb-99b6-c3d8b4107d1e.png)


## The Model

Sequential (Input shape: 64)

| Layer (type)  | Param | Trainable |
| ------------- |:-----:|----------:|
| Conv2d        | 9408  |      False|
| BatchNorm2d   | 128   |       True|
| ReLu          |       |           |
| MaxPool2d     |       |           |
| Conv2d        |  36864|      False|
| BatchNorm2d   |    128|       True|
| ReLu          |       |           |
| Conv2d        |  36864|      False|
| BatchNorm2d   |    128|       True|
| Conv2d        |  36864|      False|
| BatchNorm2d   |    128|       True|
| ReLu          |       |           |
| Conv2d        |  36864|      False|
| BatchNorm2d   |    128|       True|
Output Shape: 64 x 64 x 112 x 11

| Layer (type)  | Param | Trainable |
| ------------- |:-----:|----------:|
| Conv2d        |  73728|      False|
| BatchNorm2d   |    256|       True|
| ReLu          |       |           |
| Conv2d        | 147456|      False|
| BatchNorm2d   |    256|       True|
| Conv2d        |   8192|      False|
| BatchNorm2d   |    256|       True|
| Conv2d        | 147456|      False|
| BatchNorm2d   |    256|       True|
| ReLu          |       |           |
| Conv2d        | 147456|      False|
| BatchNorm2d   |    256|       True|
Output Shape: 64 x 128 x 28 x 28

| Layer (type)  | Param | Trainable |
| ------------- |:-----:|----------:|
| Conv2d        | 294912|      False|
| BatchNorm2d   |    512|       True|
| ReLu          |       |           |
| Conv2d        | 589824|      False|
| BatchNorm2d   |    512|       True|
| Conv2d        |  32768|      False|
| BatchNorm2d   |    512|       True|
| Conv2d        | 589824|      False|
| BatchNorm2d   |    512|       True|
| ReLu          |       |           |
| Conv2d        | 589824|      False|
| BatchNorm2d   |    512|       True|
Output Shape: 64 x 256 x 14 x 14

| Layer (type)  | Param | Trainable |
| ------------- |:-----:|----------:|
| Conv2d        |1179648|      False|
| BatchNorm2d   |   1024|       True|
| ReLu          |       |           |
| Conv2d        |2359296|      False|
| BatchNorm2d   |   1024|       True|
| Conv2d        | 131072|      False|
| BatchNorm2d   |   1024|       True|
| Conv2d        |2359296|      False|
| BatchNorm2d   |   1024|       True|
| ReLu          |       |           |
| Conv2d        |2359296|      False|
| BatchNorm2d   |   1024|       True|
Output Shape: 64 x 512 x 7 x 7

| Layer (type)  |  
| :------------ |
| AdaptiveAvgPool2d|
| AdaptiveAvgPool2d|
| Flatten   |
| BatchNorm1D (2048 params)  (Trainable)|
| Dropout   |
Output Shape: []

| Layer (type)  | Param | Trainable |
| ------------- |:-----:|----------:|
| Linear        | 524288|       True|
| ReLu          |       |           |
| BatchNorm1D   |   1024|       True|
| Dropout       |       |           |
Output: 64 x 512

| Layer (type)  | Param | Trainable |
| ------------- |:-----:|----------:|
| Linear        |   1536|       True|
Output: 64 x 3


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://github.com/AvaniBadkul/image-classification/blob/master/LICENSE)

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Author - [@AvaniBadkul](https://github.com/AvaniBadkul)

Project Link: [https://github.com/AvaniBadkul/image-classification](image-classification)







<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[build-shield]: https://img.shields.io/badge/build-passing-brightgreen.svg?style=flat-square
[build-url]: #
[license-shield]: https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square
[license-url]: https://github.com/AmiteshBadkul/image-classification-resnet18/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/amitesh-badkul
