
# OCR Model

OCR is refer to optical character recognition, the objective of this model is to make text generated based on image contain text. For this model would use CTC on CNN-RNN Based model. How this model work are slightly different for isolation method. Rather than we isolate, we split as much as we decide and then make many probabilities to each splited image

![Sample Image 1](https://raw.githubusercontent.com/Capstone-Borwita/machine-learning-path/main/OCR/Images/Ctc.png)

upper is isolation method, below is ctc method

We used research paper from Baoguang Shi, et al. in 2015 (https://arxiv.org/pdf/1507.05717) as benchmark for our architectur model.  
![Sample Image 2](https://raw.githubusercontent.com/Capstone-Borwita/machine-learning-path/main/OCR/Images/Structure.png)




## Run Locally

Clone the project

```bash
  git clone https://github.com/Capstone-Borwita/machine-learning-path.git
```

Go to the project directory

```bash
  cd OCR
```

Install dependencies

```bash
  pip install -r requirements.txt
```



## Additional information
we build our model in jupyter notebook enviroment, but also we sure can used .py extension to only used the model without training procces. This OCR model *strictly* used only cropped image contain with text, so its better used this model align with our bounding box algorithm

`Train.ipynb` : Contain how to make OCR model, this notebook can used for two model based on their font. The different only on the dataset used

`test.ipynb` : to test and compare our model with existing model.

`preprocessing.py` : to preprocessing image so it can fit in our model

`Encode-Decode.py` : contain function needed for encode and Decode
