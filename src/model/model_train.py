import argparse
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
from sklearn.model_selection import train_test_split
import datasets


def train(epochs, batch_size, data_ratio, cpu, log_step, checkpoint, save_dir):
    print("Loading models")

    torch_device = 'cuda' if torch.cuda.is_available() and not cpu else 'cpu'
    paraphrase_model_name = 'humarin/chatgpt_paraphraser_on_T5_base'

    df = pd.read_csv('data/interim/data.csv', index_col=0)
    df, _ = train_test_split(df, train_size=data_ratio, shuffle=True)

    dataset_train = datasets.Dataset.from_pandas(df)
    dataset = dataset_train.train_test_split(test_size=0.1)

    tokenizer = AutoTokenizer.from_pretrained(paraphrase_model_name)

    if checkpoint == None:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            paraphrase_model_name).to(torch_device)

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        peft_model = get_peft_model(model, peft_config)

    else:
        config = LoraConfig.from_pretrained(checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.base_model_name_or_path).to(torch_device)
        peft_model = PeftModel.from_pretrained(
            model, checkpoint, is_trainable=True).to(torch_device)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=peft_model)

    print("Preparing data")

    def preprocess_para(examples):
        prefix = 'Paraphrase: '
        model_inputs = tokenizer([prefix + example for example in examples['reference']],
                                 text_target=examples['translation'], truncation=True, padding='longest', return_tensors="pt")
        return model_inputs

    para_preprocessed = dataset.map(preprocess_para, batched=True)

    print("Training the model")

    training_args = Seq2SeqTrainingArguments(
        output_dir="tensorboard/paraphrase_finetuned",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        logging_strategy='none' if log_step == 0 else 'steps',
        num_train_epochs=epochs,
        logging_steps=log_step,
        # predict_with_generate=True,
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=para_preprocessed["train"],
        eval_dataset=para_preprocessed["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
    )

    trainer.train()

    print(f"Saving model to {save_dir}")

    peft_model.save_pretrained(save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train detoxification paraphraser')
    parser.add_argument('-e', '--epochs', type=int,
                        default=3, help="Epochs to train")
    parser.add_argument('-b', '--batch-size', type=int,
                        default=16, help='Batch size')
    parser.add_argument('--subset-ratio', type=float, default=0.15,
                        help='How much data of the dataset use to train. Should bi in (0,1]')
    parser.add_argument('--cpu', action="store_true",
                        help="Use CPU for computations. Otherwise, CUDA device will be used if found")
    parser.add_argument('--log-step', type=int, default=10,
                        help='How frequent to print training logs. If set to 0, no logs will be printed')
    parser.add_argument('--checkpoint', type=str,
                        help='Checkpoint to use for the baseline. If not chosen, the model will be trained from scratch. If you want to train already fine-tuned model, use \"model/para_ft_2\"')
    parser.add_argument('--save-dir', type=str,
                        help='Directory to save the trained model')
    args = parser.parse_args()

    train(args.epochs, args.batch_size, args.subset_ratio,
          args.cpu, args.log_step, args.checkpoint, args.save_dir)
