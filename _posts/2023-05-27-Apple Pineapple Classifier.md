---
jupyter: python3
---

This is the code I used for making a classifier between apples and pineapples.

It uses a resnet18 architecture and is trained on 100 images of apples and 100 pineapples.

It is pretrained with fast ai's vision learner

```{python}
#| id: efehmJnAIqpR
#| id: efehmJnAIqpR
!pip install -Uqq fastai
!pip install -Uqq bing_image_downloader
from fastai.vision.all import *
```

```{python}
#| id: 8lW0DPDZbmkD
#| colab: {base_uri: 'https://localhost:8080/'}
#| id: 8lW0DPDZbmkD
#| outputId: 06128002-f4de-46e4-ae4d-a6678427719b
!pip install -Uqq gradio
import gradio as gr
```

```{python}
#| id: ODqMRjFVVQNn
#| colab: {base_uri: 'https://localhost:8080/'}
#| id: ODqMRjFVVQNn
#| outputId: 806071c3-d933-4833-bbcc-dfcb95ad127c
from google.colab import drive
drive.mount('/content/drive')
```

Downloads the training data using bing image downloader api

```{python}
#| id: 5SOyvu-TJysT
#| id: 5SOyvu-TJysT
from bing_image_downloader import downloader

downloader.download("apple fruit", limit=100,  output_dir='dataset', adult_filter_off=True, force_replace=False, timeout=60)
downloader.download("pineapple fruit", limit=100,  output_dir='dataset', adult_filter_off=True, force_replace=False, timeout=60)
```

```{python}
#| id: bYVqjx3jhVwN
#| colab: {base_uri: 'https://localhost:8080/', height: 752}
#| id: bYVqjx3jhVwN
#| outputId: 1d19cbc7-fdef-455f-dd8d-ec6135e2b469
path = Path('dataset')
dls = ImageDataLoaders.from_folder(path, train=".", valid_pct=0.2, seed=42, item_tfms=Resize(224))

# Checking the data
dls.show_batch(nrows=3, ncols=3)
```

```{python}
#| id: JtuZqpKM5PEP
#| colab: {base_uri: 'https://localhost:8080/', height: 334}
#| id: JtuZqpKM5PEP
#| outputId: 982cc380-5a22-45ba-b436-582d7e8cfaa3
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```

```{python}
#| id: I8UYuPUF8ifw
#| id: I8UYuPUF8ifw
learn.export('model.pkl')
```

```{python}
#| id: f-bMPagtYy8O
#| id: f-bMPagtYy8O
!mv 'dataset/model.pkl' '/content/drive/My Drive/Colab Projects/Apple-Pineapple'
```

```{python}
#| id: GHgmHIhTGWcJ
#| id: GHgmHIhTGWcJ
learn = load_learner('drive/My Drive/Colab Projects/Apple-Pineapple/model.pkl')
```

```{python}
#| id: elfq-UzPGZFo
#| id: elfq-UzPGZFo
labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}
```

```{python}
#| id: GTg5HaEmbZqy
#| colab: {base_uri: 'https://localhost:8080/', height: 755}
#| id: GTg5HaEmbZqy
#| outputId: c109a4bf-4b2e-4fe1-d9cf-60e9de94ba64

gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(512, 512)), outputs=gr.outputs.Label(num_top_classes=3)).launch(share=True)
```


