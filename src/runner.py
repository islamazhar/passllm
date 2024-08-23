
import sys 
import os 
import random 
import torch 

import numpy as np

def parse_args():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--output_dir", default="../log/finetune_models/", type=str, help="direction of the fine tune model location")
    parser.add_argument("--datafile", default="../log/", type=str, help="direction of the JSON datafile to fine tune the model")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    return args 
    

def quantize_model(model_name):
    """ Load the model and return the quantized model """
    
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map={"":0}
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    return model, tokenizer 


def get_data(datafile, size = None):
    """ read the datafile (must be in JSON format)"""
    from datasets import load_dataset
    dataset = load_dataset('json', data_files=datafile, split='train')
    
    if size is not None and size < len(dataset):
        shuffled_dataset = dataset.shuffle(seed=42)
        sampled_dataset = shuffled_dataset.select(range(size))
        return sampled_dataset
        
    return dataset

def finetune(model, tokenizer, dataset, outdir):
    peft_params = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_params = TrainingArguments(
        output_dir=outdir,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        logging_steps=1,
        learning_rate=2e-4,
        fp16=True
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_params,
        dataset_text_field="inputs",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )
    
    trainer.train()
    
    
    trainer.model.save_pretrained(outdir)
    trainer.tokenizer.save_pretrained(outdir)
    
    # return model_name
    
def inference(nmodel, bmodel):
    
    # 1. Load based model
    base_model = AutoModelForCausalLM.from_pretrained(bmodel,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
    )
    
    # 2. Load new fine-tuned model. Then merge this two model
    finetuned_model = PeftModel.from_pretrained(base_model, nmodel)
    merged_model = finetuned_model.merge_and_unload()
    
    # 3. Load tokenizer,
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 4. Define pipeline
    pipe = pipeline(task="text-generation", model=merged_model, tokenizer=tokenizer, max_length=2000)
    
    prompt_template = PromptTemplate(
    input_variables=["instruction", "passwords_chosen", "question", "answer"], template="<s>[INST] <<SYS>>{instruction}. Here is the list  (separated by '\t') {pw_others}<</SYS>>{question}[/INST]"
    )
    
    pw_others = "mauser"
    input = prompt_template.format(instruction=instruction, pw_others = pw_others, question = question)
    
    result = pipe(input)
    
    print(result[0]['generated_text'].split("/INST")[1][1:])
        
    
    
    
    
def main():
    args = parse_args()
    qmodel, tokenizer  = quantize_model(args.model)
    dataset = get_data(args.datafile)
    nmodel = finetune(qmodel, tokenizer, dataset, args.output_dir)
    predict = inference(nmodel, args.model)
    
    

    
if __name__ == "__main__":
    print(sys.argv)
    main()
    
    

# python3 runner.py --model meta-llama/Llama-2-7b-chat-hf --output_dir location where the fine tuned model will be saved --datafile 
