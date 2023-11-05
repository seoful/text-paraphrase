import torch
import numpy as np
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
import argparse
import warnings
warnings.filterwarnings("ignore")


class CosSimilarityPipeline():
    def __init__(self, model) -> None:
        self.model = model

    def _forward(self, inputs):
        embeds = self.model.encode(inputs)
        return embeds

    def _postprocess(self, outputs_inputs, outputs_targets):
        s = []
        for input_embd, target_embd in zip(outputs_inputs, outputs_targets):
            cos_sim = torch.nn.functional.cosine_similarity(
                torch.tensor(input_embd), torch.tensor(target_embd), dim=0)
            s.append(cos_sim)
        return torch.tensor(s)

    def __call__(self, inputs, targets, reduce_mean=True):
        input_embeds = self._forward(inputs)
        targets_embeds = self._forward(targets)
        sims = self._postprocess(input_embeds, targets_embeds)
        if reduce_mean:
            return torch.mean(sims).item()
        else:
            return sims.numpy()


def paraphrase(
    question,
    model,
    tokenizer,
    device,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    tokenized = tokenizer(
        f'Paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    )
    tokenized = tokenized.to(device)

    outputs = model.generate(
        **tokenized, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res


def detoxify(text, model, tokenizer, device, toxicity_evaluator_pipeline, cos_sim_pipeline, num_generate_sequences=10, return_metrics=True):
    with torch.no_grad():
        paraphrased_texts = paraphrase("Paraphrase: " + text, model, tokenizer, device, num_return_sequences=num_generate_sequences,
                                       num_beams=5 if num_generate_sequences == 1 else num_generate_sequences)
        toxicities = toxicity_evaluator_pipeline(paraphrased_texts)
        postprocessed_detoxicities = []
        for row in toxicities:
            if row['label'] == 'neutral':
                postprocessed_detoxicities.append(row['score'])
            else:
                postprocessed_detoxicities.append(1-row['score'])

        similarities = cos_sim_pipeline(
            paraphrased_texts, [text]*num_generate_sequences, reduce_mean=False)

        if num_generate_sequences > 1:
            means = np.mean(
                np.stack([postprocessed_detoxicities, similarities], axis=1), axis=1)
            best_id = np.argmax(means)
            output = paraphrased_texts[best_id]
            metrics = {
                'toxicity': 1 - postprocessed_detoxicities[best_id], 'similarity': similarities[best_id]}
        else:
            output = paraphrased_texts[0]
            metrics = {
                'toxicity': 1 - postprocessed_detoxicities[0], 'similarity': similarities[0]}

    if return_metrics:
        return output, metrics
    else:
        return output


def predict(text, num_return_sequences, cpu, print_metrics, checkpoint):

    torch_device = 'cuda' if torch.cuda.is_available() and not cpu else 'cpu'

    toxicity_classifier_name = 'SkolkovoInstitute/roberta_toxicity_classifier'
    similarity_embedder_name = 'sentence-transformers/all-MiniLM-L6-v2'

    print("Loading models")

    config = PeftConfig.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, checkpoint).to(torch_device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    toxicity_clf_tokenizer = AutoTokenizer.from_pretrained(
        toxicity_classifier_name)
    toxicity_clf_model = AutoModelForSequenceClassification.from_pretrained(
        toxicity_classifier_name)
    toxicity_clf_model.eval()
    toxicity_clf_pipeline = pipeline(
        'text-classification', tokenizer=toxicity_clf_tokenizer, model=toxicity_clf_model, device=torch_device)

    sentence_similarity_model = SentenceTransformer(similarity_embedder_name)
    sentence_similarity_model.eval()
    cos_similarit_pipeline = CosSimilarityPipeline(sentence_similarity_model)

    result = detoxify(text, model, tokenizer, torch_device, toxicity_clf_pipeline,
                      cos_similarit_pipeline, num_return_sequences, print_metrics)

    if print_metrics:
        print(result[0])
        print(result[1])
    else:
        print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detoxification')
    parser.add_argument('text', type=str, help='Text to detoxify')
    parser.add_argument('--cpu', action="store_true",
                        help="Use CPU for computations. Otherwise, CUDA device will be used if found")
    parser.add_argument('-n', '--num-generated-shots', type=int, default=10,
                        help="How many sequences will be generated to choose from at an evaluation step")
    parser.add_argument('-p', '--print-metrics', action='store_false',
                        help='Print metrics of the generated text')
    parser.add_argument('-m', '--model-checkpoint', type=str, default='model/para_ft_2',
                        help='Directory of the model checkpoint to use. Defaults to alredy pretrained one')
    args = parser.parse_args()
    print(args)
    predict(args.text, args.num_generated_shots, args.cpu,
            args.print_metrics, args.model_checkpoint)
