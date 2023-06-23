 # Authors: Siddharth Suresh <siddharth.suresh@wisc.edu>
 #          Alex Huang <whuang288@wisc.edu>
 #          Xizheng Yu <xyu354@wisc.edu>
 
import csv
import time
import yaml
from time import sleep
from pathlib import Path
from joblib import Parallel, delayed

import torch
import openai
import transformers
from accelerate import Accelerator

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

config['model_list'] = config['model_pipeline'] + config['model_decode'] + config['gpts'] + config['gpts_deployment']

def timer(start_time):
    """the helper function for running time"""
    minutes, seconds = divmod(time.time() - start_time, 60)
    return f"{int(minutes)} mins {int(seconds)} sec"

def write_responses(res, file_path, mode):
    """the helper function for saving the responses"""
    # mode: 'w', 'a'
    with open(file_path, mode) as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(res)

def send_gpt_prompt(batch, model_type, prompt_and_response, temperature, cot, max_tokens):
    """helper function to send a whole to chatgpt"""
    for prompt in batch:
        succeed = False
        completion = None
        while not succeed:
            try:
                completion = openai.Completion.create(
                    engine = model_type,
                    prompt = prompt,
                    max_tokens = max_tokens,
                    n = 1,
                    temperature = temperature,
                )
                succeed = True
            except Exception as e:
                print("GPT sleeping...")
                sleep(60)
        assert completion is not None
        response = completion['choices'][0]['text'].replace('\n', ' ').replace(' .', '.').strip()
        if cot == True:
            prompt_and_response.append([prompt, response, response.split(" ")[-1][:-1]])
        else:
            prompt_and_response.append([prompt, response])
        
def generate_responses_gpt(batches, model_type, output_path, temperature, cot, max_tokens):
    start_time = time.time()
    
    openai.api_key = Path(f"AZURE_OPENAI_KEY").read_text()
    
    if config['openai_azure']:
        openai.api_base = Path(f"AZURE_OPENAI_ENDPOINT").read_text()
        openai.api_type = 'azure'
        openai.api_version = '2023-05-15'
        
    prompt_and_response = []
    
    # can change n_jobs accoding to the size of dataset
    Parallel(n_jobs = 10, require='sharedmem')(
        delayed(send_gpt_prompt)(
            batch, model_type, prompt_and_response, temperature, cot, max_tokens
        ) for batch in batches
    )
    print(f'Time taken to generate responses is {timer(start_time)}s')
    
    write_responses(prompt_and_response, output_path, 'w')
    
    return prompt_and_response

def load_model(model_type):
    """Load the model and tokenizer based on the model type."""
    
    if model_type == "llama-7b":
        model = transformers.LlamaForCausalLM.from_pretrained("/mnt/disk-1/llama_hf_7B")
        tokenizer = transformers.LlamaTokenizer.from_pretrained("/mnt/disk-1/llama_hf_7B")
    elif model_type == "alpaca-7b":
        model = transformers.AutoModelForCausalLM.from_pretrained("/mnt/disk-1/alpaca-7b")
        tokenizer = transformers.AutoTokenizer.from_pretrained("/mnt/disk-1/alpaca-7b")
    elif model_type == "flan-ul2":
        model = transformers.T5ForConditionalGeneration.from_pretrained("google/flan-ul2", torch_dtype=torch.bfloat16, cache_dir="/mnt/disk-1/flan-ul2")         
        tokenizer = transformers.AutoTokenizer.from_pretrained("google/flan-ul2")
    elif model_type == "flan-t5-xl":
        model = transformers.T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", torch_dtype=torch.bfloat16, cache_dir="/mnt/disk-1/flan-t5-xl")         
        tokenizer = transformers.AutoTokenizer.from_pretrained("google/flan-t5-xl")
    elif model_type == "flan-t5-xxl":
        model = transformers.T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", torch_dtype=torch.bfloat16, cache_dir="/mnt/disk-1/flan-t5-xxl")         
        tokenizer = transformers.AutoTokenizer.from_pretrained("google/flan-t5-xxl")
    elif model_type == "falcon-7b":
        model = transformers.AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", torch_dtype=torch.bfloat16, cache_dir="/mnt/disk-1/falcon-7b", trust_remote_code=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained("tiiuae/falcon-7b")  

    return model, tokenizer

def initialize_pipeline(model, tokenizer, device):
    """Initialize the pipeline."""
    pipe = transformers.pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    return pipe

def generate_responses_pipeline(batch, pipe):
    """Generate responses for a batch for models that use a pipeline."""
    responses = pipe(batch, max_new_tokens=20, do_sample=False)
    responses = [r[0]['generated_text'] for r in responses]
    responses = [','.join(r.split('\n')[1::]) for r in responses]
    return responses

def generate_responses_decode(batch, model, tokenizer, device, max_length=256):
    """Generate responses for a batch for models that need decoding."""
    # prepare and tokenize inputs
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # generate response
    outputs = model.generate(**inputs, max_length=max_length, do_sample=False)
    # decode outputs
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return responses

def get_responses(batches, model_type, temperature, output_path, cot, batch_size=256, max_tokens=256):
    if model_type not in config['model_list']:
        print(f"Error: Please select from following models: {config['model_list']}")
        exit()
        
    if model_type in config['gpts'] or model_type in config['gpts_deployment']:
        return generate_responses_gpt(batches, model_type, output_path, temperature, cot, max_tokens)
        
    batches = batches[0]

    start_time = time.time()
    transformers.logging.set_verbosity_error()
    
    accelerator = Accelerator()
    device = accelerator.device
    model, tokenizer = load_model(model_type)
    model, tokenizer = accelerator.prepare(model, tokenizer)
    
    if model_type in config['model_pipeline']:
        # pipeline for text generation
        pipe = initialize_pipeline(model, tokenizer, device)
        
    print(f'Time taken to load model is {timer(start_time)}.')
    
    # all_responses = []
    for i in range(0, len(batches), batch_size):
        batch = batches[i:i+batch_size]
        if model_type in config['model_pipeline']:
            responses = generate_responses_pipeline(batch, pipe)
        elif model_type in config['model_decode']:
            responses = generate_responses_decode(batch, model, tokenizer, device)
        # all_responses.extend(responses)
        if cot == True:
            answers = [r.split(" ")[-1][:-1] for r in responses] # retreive the answer
            write_responses(list(zip(batch, responses, answers)), output_path, 'a') # write this batch
        else:
            write_responses(list(zip(batch, responses)), output_path, 'a') # write this batch
        print(f'Time taken to generate responses for all previous batches is {timer(start_time)}.')

    
    prompt_and_response = [] # list(zip(batches, all_responses))
    print(f'Time taken to generate all responses is {timer(start_time)}.')
    return prompt_and_response

# # interactions with transformer
# def get_transformer_responses(batches, model_type, model_name, batch_size):
    
#     responses = []
#     prompt_and_response = [] 
#     batches = np.array(list(itertools.chain(*batches)))
#     start_time = time.time()
    
#     # for gpu computing
#     accelerator = Accelerator()
#     device = accelerator.device
#     tokenizer = accelerator.prepare(
#         T5Tokenizer.from_pretrained(model_name)
#     )
    
#     # prepare the dataset
#     prompt_list = batches.tolist()
#     max_length = max([len(prompt.split()) for prompt in prompt_list])
#     prompt_dict = {'prompt':prompt_list}
    
#     ds = Dataset.from_dict(prompt_dict)
#     ds = ds.map(lambda examples: T5Tokenizer.from_pretrained(model_name)(examples['prompt'], max_length=max_length, truncation=True, padding='max_length'), batched=True)
#     ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
#     dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    
#     flan_model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16,  cache_dir="./models")
#     flan_model = flan_model.to(device)
    
#     # get the responses
#     preds = []
#     for batch in dataloader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         # print(f"input: {input_ids}")
#         # print(f"attention_mask: {attention_mask}")
#         outputs = flan_model.generate(input_ids, attention_mask=attention_mask, renormalize_logits = True)
#         preds.extend(outputs)
    
#     responses = tokenizer.batch_decode(preds, skip_special_tokens=True)
#     del flan_model
#     # return the results
#     for prompt, response in zip(batches, responses):
#         prompt_and_response.append([prompt, response])
        
#     print('Time taken to generate responses is {}s'.format(time.time()-start_time))
#     return prompt_and_response