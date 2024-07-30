# Color-Extraction
Prerequisite: <br>
1. ```pip install transformers==4.42.3```
2. ```pip install accelerate==0.32.1 -U```
3. ```pip install datasets```

Goal: To extract the color words from a product name.
## color_bert_train.py
This file fine-tunes [the model](<https://huggingface.co/Babelscape/wikineural-multilingual-ner>) from HuggingFace.
## color_bert_test.py
This file demonstrates how to use the tuned model directly and the results.
