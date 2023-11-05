# Creation process

## Approach 1: Mask-fill approach
The simplest way to do the detoxification task is using mask-fill approach. the idea is using BERT key feature: masking some tokens and predicting them to form the embeddings. It can be used in a way that we encode toxic words as ```<mask>``` and let the model predict them to form more formal-style text. The downside of this approach is that we firstly need to classify each word in the sentence for toxicity, therefore train an additional model using external dataset. Moreover, studies show the superiority of using LLM text2text models over mask-fill approach. So, I did not stick with this approach. 

## Approach 2: Pre-trained model for paraphrasing
On hugging face I found pre-trained pegasus model fine-tuned for paraphrasing ([link](https://huggingface.co/tuner007/pegasus_paraphrase)) which does quite a good job on hugging face inference API. It even tries to detoxify text from the box. However, it is very heavy to trai. So, I chose the analogous paraphrasing model based on T5 ([link](https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base?text=When+should+I+go+to+the+doctor%3F))

## Metrics
In order to evaluate the performance of the model, I needed some metrics to see the accuracy of the model (how it performs the detoxification) and the semantic difference between original and generated texts. So, I found a fine-tuned roberta model for the toxicity classification ([link](https://huggingface.co/s-nlp/roberta_toxicity_classifier?text=Some+weights+of+the+model+checkpoint+at++fuck)) and embedder model to track the cosine similarity ([link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)).

## Approach 3: The chain of models
While searching for the models to track metrics, I thought about using these models as the second step of the detoxifying chain. The idea is that the model discussed in Appr.3 can produce several different outputs for the same input. So, we can generate for example 10 outputs and calculate their toxicity levels and cosine similarities with the reference texts to choose the best one. Hopefully, it will give more appropriate result due to more variability. Also, we can fine-tune the toxicity classifier model to the levels of toxicity marked by hands we have in our dataset. 

## Long training process
Since LLM are hard to train in any case, I decided to use only the subset of the original dataset to make the training faster. However, it makes model less intelligent. But still, there is a possibility to train it on the whole dataset.

## Result
I fine-tuned pretrained T5 model from appr.2 on the subset of the original dataset. Then it generetes several shots at inference and with the usage of models from Metrics, I chose the best of them by the best average results.