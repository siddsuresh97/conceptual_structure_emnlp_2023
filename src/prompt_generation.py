 # Authors: Siddharth Suresh <siddharth.suresh@wisc.edu>
 #          Alex Huang <whuang288@wisc.edu>
 #          Xizheng Yu <xyu354@wisc.edu>
 
import numpy as np
import logging

ERROR = 3
ESTIMATED_RESPONSE_TOKENS = 8 + ERROR

def generate_prompt(exp, input, CoT):
    """General function generating prompts for different experiments"""
    
    if exp == 'pairwise':
        a, b = input
        # prompt = f"Answer with only one number from 1 to 5, considering 1 as 'very dissimilar', 2 as 'likely dissimilar', 3 as 'neutral', 4 as 'likely similar', and 5 as 'very similar': How similar is {a} and {b}?"
        # prompt = f'Answer with only one number from 1 to 7: How similar is {a} and {b}, considering 1 as the least similar and 7 as the most similar?'
        prompt = f"Answer with only one number from 1 to 7, considering 1 as 'extremely dissimilar', 2 as 'very dissimilar', 3 as 'likely dissimilar', 4 as 'neutral', 5 as 'likely similar', 6 as 'very similar', and 7 as 'extremely similar': How similar is {a} and {b}?"
    elif exp == 'triplet':
        flipped = False
        anchor, concept1, concept2 = input
        if flipped:
            concept1, concept2 = concept2, concept1
        prompt = f"Answer using only one word - {concept1} or {concept2} and not {anchor}. Which is more similar in meaning to {anchor}?"
        # prompt = f'Answer using only one word - {concept1} or {concept2} and not {anchor}, which is more similar to {anchor} in terms of size?'
    elif exp == 'feature':
        prompt = f'Generate an grammatically correct English prompt that checks if [{input}] is true for an object. Use [placeholder] to represent the object.'
    elif exp == 'q_and_a':
        pass
    
    if CoT == True:
        prompt = prompt + " End your answer with, 'The answer is '. Let's think step by step. "
    
    characters = len(prompt)

    return prompt, characters

def make_prompt_batches(exp, inputs, CoT):
    total_tokens = 0
    batches = []
    batch = []
    for input in inputs:
        prompt, characters = generate_prompt(exp, input, CoT)
        tokens = np.ceil((characters + 1)/4)
        if total_tokens < 100000:
            batch.append(prompt)
            total_tokens = tokens + (ESTIMATED_RESPONSE_TOKENS)
        else:
            batches.append(batch)
            batch = [prompt]
            total_tokens = tokens + (ESTIMATED_RESPONSE_TOKENS)
    if len(batch) != 0:
        batches.append(batch)
    logging.info('Total batches of 100000 tokesn are {}'.format(len(batches)))
    return batches

# generating prompts for triplet experiments
# def generate_prompt_triplet(anchor, concept1, concept2):
#     prompt = 'Answer using only only word - {} or {} and not {}. Which is more similar in meaning to {}?'.format(concept1, concept2, anchor, anchor)
#     characters = len(prompt)
#     return prompt, characters

# def make_prompt_batches_triplet(triplets):
#     total_tokens = 0
#     batches = []
#     batch = []
#     for triplet in triplets:
#         anchor, concept1, concept2 = triplet
#         prompt, characters = generate_prompt_triplet(anchor, concept1, concept2)
#         tokens = np.ceil((characters + 1)/4)
#         if total_tokens < 100000:
#             batch.append(prompt)
#             total_tokens = tokens + (ESTIMATED_RESPONSE_TOKENS)
#         else:
#             batches.append(batch)
#             batch = [prompt]
#             total_tokens = tokens + (ESTIMATED_RESPONSE_TOKENS)
#     if len(batch) != 0:
#         batches.append(batch)
#     logging.info('Total batches of 150000 tokesn are {}'.format(len(batches)))
#     return batches
 
# generating prompts for the general Q&A experiments
def make_prompt_batches_q_and_a(prompts):
    total_tokens = 0
    batches = []
    batch = []
    for prompt in prompts:
        characters = len(prompt)
        tokens = np.ceil((characters + 1)/4)
        if total_tokens < 100000:
            batch.append(prompt)
            total_tokens = tokens + (ESTIMATED_RESPONSE_TOKENS)
        else:
            batches.append(batch)
            batch.append(prompt)
            total_tokens = tokens + (ESTIMATED_RESPONSE_TOKENS) 
    if len(batch) != 0:
        batches.append(batch)
    logging.info('Total batches of 100000 tokesn are {}'.format(len(batches)))
    return batches

# # generating prompts for the "concept and feature" experiment
# def generate_prompt_feature(feature):
#     prompt = f'Generate an grammatically correct English prompt that checks if [{feature}] is true for an object. Use [placeholder] to represent the object.'
#     characters = len(prompt)
#     return prompt, characters

# def make_prompt_batches_feature(features):
#     total_tokens = 0
#     batches = []
#     batch = []
#     for feature in features:
#         prompt, characters = generate_prompt_feature(feature)
#         tokens = np.ceil((characters + 1)/4)
#         if total_tokens < 100000:
#             batch.append(prompt)
#             total_tokens = tokens + (ESTIMATED_RESPONSE_TOKENS)
#         else:
#             batches.append(batch)
#             batch = [prompt]
#             total_tokens = tokens + (ESTIMATED_RESPONSE_TOKENS)
#     if len(batch) != 0:
#         batches.append(batch)
#     logging.info('Total batches of 100000 tokesn are {}'.format(len(batches)))
#     return batches