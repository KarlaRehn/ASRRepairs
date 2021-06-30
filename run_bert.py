
import datasets
from datasets import load_metric
import numpy as np
from transformers import AutoTokenizer, BertTokenizerFast
from transformers import EncoderDecoderModel, BertGenerationEncoder, BertGenerationDecoder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from typing import Dict

class LoggingTrainer(Seq2SeqTrainer):
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

        logSave = open("logfile_bert2bert_seq_NSTtrain.txt", 'a')
        logSave.write(str(output) + '\n')
        logSave.close()


def process_data_to_model_inputs(batch, enc_max_len = 32, dec_max_len = 32):
  # tokenize the inputs and labels
  asr_lower = [asr.lower() for asr in batch["ASR"]]
  gt_lower = [gt.lower() for gt in batch["GT"]]
  inputs = tokenizer(asr_lower, padding="max_length", truncation=True, max_length=enc_max_len)
  outputs = tokenizer(gt_lower, padding="max_length", truncation=True, max_length=dec_max_len)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
  # We have to make sure that the PAD token is ignored
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch

def split_and_process_data(raw_data, n_val_samples = 1000, batch_size = 16 ):
	raw_data = raw_data.shuffle()
	#n_val_samples = 1000 #change to 100 for ville and SC
	n = raw_data.num_rows['train'] - n_val_samples
	training_samples = range(n)
	val_samples = [x + n for x in range(n_val_samples)]
	train_data = raw_data['train'].select(training_samples)



	train_data = train_data.map(
    		process_data_to_model_inputs, 
    		batched=True, 
    		batch_size=batch_size, 
    		remove_columns=["GT", "ASR"] # Removing the raw texts, don't need those anymore
	)

	train_data.set_format(
    		type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
	)

	val_data = raw_data['train'].select(val_samples)
	val_data = val_data.map(
    		process_data_to_model_inputs, 
    		batched=True, 
    		batch_size=batch_size, 
    		remove_columns=["GT", "ASR"]
	)

	val_data.set_format(
    		type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
	)
	
	return train_data, val_data
	
	metric = load_metric("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

	

torch.cuda.empty_cache()
raw_data = datasets.load_dataset('json', data_files={'train':['authentic_trainsets/tri2b_NSTClean_train.json']})
train_data, val_data = split_and_process_data(raw_data)

tokenizer = BertTokenizerFast.from_pretrained("KB/bert-base-swedish-cased")

#bert_point = "KB/bert-base-swedish-cased"
#bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(bert_point, bert_point)

bert_point = "bert2bert_gen1"
bert2bert = EncoderDecoderModel.from_pretrained(bert_point)


bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
bert2bert.config.eos_token_id = tokenizer.sep_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id
bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size
bert2bert.config.max_length = 32
bert2bert.config.min_length = 1
bert2bert.config.no_repeat_ngram_size = 3
bert2bert.config.early_stopping = True
bert2bert.config.length_penalty = 2.0
bert2bert.config.num_beams = 4

# To freeze the weights in the encoder
#for param in bert2bert.encoder.parameters():
#        param.requires_grad = False

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True, 
    output_dir="./BERT2BERT",
     logging_steps=1000,
     save_steps=30000,
     eval_steps=1000,
     warmup_steps=1000,
     save_total_limit=3,
     num_train_epochs=5
)


# See what the untrained model says
#input_ids = tokenizer("Hej", return_tensors="pt").input_ids
#greedy_output = bert2bert.generate(input_ids, max_length=50)
#print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

trainer = LoggingTrainer(
    model=bert2bert,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()

bert2bert.save_pretrained('bert2bert')

trainer.evaluate()

# See what the trained model says
#input_ids = tokenizer("FOCKENS hÅN LEDER HONOM FRoM TILL Vätten OCH BÖRJE VILL RÖRA vid det".lower(), return_tensors="pt").input_ids
#greedy_output = bert2bert.generate(input_ids, max_length=50)
#print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

