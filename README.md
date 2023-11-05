# Text detoxification
> Agafonov Alexander

> a.agafonov@innopolis.university

> B21-DS-01

## Model
The detoxification framework is based on the following algorithm:

* Firstly, paraphrasing model generates specified number of paraphrased texts (e.g. 10).

* For each of the generated texts, toxicity and cosine similarity with the original prompt are calculated.

* Text with the highest average of two metrics is chosen.

More information can be read in the ```reports``` folder.

## Data
If you want to train the model or use the dataset, firstly run the foolowing command to create dataset file from shards.

```python src/data/create_data.py```

## Training
The model is already trained on the detoxification dataset. However, it is possible to train it from scratch or train it more from the stored checkpoint

To train the model run the following command:
```python src/model/model_train.py --checkpoint /model/para_ft_2 --save-dir <DIR_TO_SAVE_TRAINED_MODEL>```

To train the model from scratch use 
```python src/model/model_train.py --save-dir <DIR_TO_SAVE_TRAINED_MODEL>```



Arguments for the command
```
optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        Epochs to train
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
  --subset-ratio SUBSET_RATIO
                        How much data of the dataset use to train. Should bi in (0,1]
  --cpu                 Use CPU for computations. Otherwise, CUDA device will be used if found
  --log-step LOG_STEP   How frequent to print training logs. If set to 0, no logs will be printed
  --checkpoint CHECKPOINT
                        Checkpoint to use for the baseline. If not chosen, the model will be trained from scratch. If
                        you want to train already fine-tuned model, use "model/para_ft_2"
  --save-dir SAVE_DIR   Directory to save the trained model
```

## Inference
To use the trained model run this command

```python src/model/model_predict.py "<YOUR_TEXT_HERE>" -n 10 -p```


Here are the arguments for the predict command
```
positional arguments:
  text                  Text to detoxify

optional arguments:
  -h, --help            show this help message and exit
  --cpu                 Use CPU for computations. Otherwise, CUDA device will be used if it is found
  -n, --num-generated-shots
                        How many sequences will be generated to choose from at an evaluation step
  -p, --print-metrics   Print metrics of the generated text
  -m, --model-checkpoint
                        Paraphrase model checkpoint to use. Defaults to alredy pretrained one
```