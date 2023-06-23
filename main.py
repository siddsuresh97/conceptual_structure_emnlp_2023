 #################################################################################
 # The code is originally written by Siddharth Suresh (siddharth.suresh@wisc.edu)#
 # Repurposed and modified by Alex Huang (whuang288@wisc.edu)                    #
 #################################################################################


import argparse, os
import logging
import csv
import pandas as pd
import numpy as np
from src.model_interaction import *
from src.prompt_generation import *

def save_responses(reponses, file_path, mode):
    """the helper function for saving the responses"""
    # mode: 'w', 'a'
    with open(file_path, mode) as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(reponses)
    
def run_exp(exp_name,  
            model_type,
            input_path,
            output_path, 
            batch_size=256, 
            max_tokens=256,
            temperature=0,
            cot=False
            ):
    """the helper function for running the experiment"""
    
    # get the batches accodring to the experiment type
    if exp_name == 'triplet' or exp_name == 'pairwise':
        assert len(input_path) == 1, "Triplet or Pairwise Experiment should only have one input file"
        input_file = np.loadtxt(input_path[0], delimiter=',', dtype = str)
        batches = make_prompt_batches(exp_name, input_file, cot)
    elif exp_name == 'q_and_a':
        assert len(input_path) == 1, "Q&A Experiment should only have one input file"
        input_file = np.loadtxt(input_path[0], delimiter=',', dtype = str)
        batches = make_prompt_batches_q_and_a(input_file)
    elif exp_name == 'feature_and_concept':
        assert len(input_path) == 2, "Feature and Concept Experiment should have two input files"
        feature_file = np.loadtxt(input_path[0], delimiter='\n', dtype = str)
        batches = make_prompt_batches(exp_name, feature_file, cot)
    else:
        logging.error('Undefined task. Only feature listing and triplet implemented')
    
    # print out info about this run
    print('Running experiment {} on data {} using {} model. Please wait for it to finish'.format(exp_name, input_path, model_type))
    
    responses = get_responses(batches, model_type, temperature, output_path, cot, batch_size)
        
    # pipeline specific for the Feature and Concept experiment
    if exp_name == "feature_and_concept":
        with open( output_path, 'w') as output_file:
            with open(input_path[1], 'r') as concept_file:
                concepts = concept_file.readlines()
                for concept in concepts:
                    concept = concept.strip("\n")
                    for _, prompt in responses:
                        output_file.write(prompt.replace("[placeholder]", concept) + "\n")
    # else:
    #     save_responses(responses, output_path, 'w')

def main():
    """the main method, handling command line arguments"""
    
    # parse the arguments
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument('--exp_name', 
                        default=None,
                        type=str, 
                        help="""the experiment type that you are doing""")
    parser.add_argument('--model_type', 
                        default=None,
                        type=str, 
                        help="""Please select from following models: 'flan-t5-xxl, flan-t5-xl, flan-ul2, llama-7b, alpaca-7b, falcon-7b, and gpt families""")
    parser.add_argument('--input', 
                        default=[], 
                        nargs='*',
                        help="""path to the input file""")
    parser.add_argument('--output', 
                        type=str, 
                        default=None,
                        help="""path to the output file""")
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=256, 
                        help="""The batch size of data that is fed to the LLM""")
    parser.add_argument('--temperature', 
                        type=float, 
                        default=0, 
                        help="""Temperature for LLMs""")
    parser.add_argument('--cot', 
                        type=bool, 
                        default=False, 
                        help="""Running Chain of Thought for your experiment""")
    args = parser.parse_args()
    
    # check if arguments was provided
    assert args.exp_name != None
    assert args.model_type != None
    assert len(args.input) != 0
    assert args.output != None
    
    
    # log the info to the log file
    os.makedirs("logs/", exist_ok=True)
    logging.basicConfig(filename="logs/{}_{}.log".format(args.exp_name, args.model_type), level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.warning('is when this event was logged.')
    logging.info('Running experiments with the following parameters')
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)
    
    # call the helper function to do the actual work
    run_exp(exp_name = args.exp_name,  
            model_type = args.model_type, 
            input_path = args.input,
            output_path = args.output, 
            batch_size = args.batch_size,
            max_tokens = 256,
            temperature = args.temperature,
            cot = args.cot,)        

if __name__=="__main__":
    main()
