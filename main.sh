# CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './experiments/triplet/prompt.csv' --output './experiments/triplet/response.csv' --batch_size 256
# python main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './experiments/triplet/prompt.csv' --output './experiments/triplet/response.csv' --batch_size 128

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './experiments/triplet/prompt.csv' --output './experiments/triplet/response.csv' --batch_size 128
# python main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './experiments/triplet/prompt.csv' --output './experiments/triplet/response.csv' --batch_size 256

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --exp_name 'q_and_a' --model_type 'flan' --model_name 'google/flan-t5-xxl' --input './experiments/q_and_a/prompt.csv' --output './experiments/q_and_a/response.csv' --batch_size 256
# python main.py --exp_name 'q_and_a' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './experiments/q_and_a/prompt.csv' --output './experiments/q_and_a/response.csv' --batch_size 256

# python main.py --exp_name 'triplet' --model_type 'gpt' --model_name 'text-davinci-003' --input './experiments/triplet/prompt.csv' --output './experiments/triplet/response.csv' --batch_size 256
# python main.py --exp_name 'q_and_a' --model_type 'gpt' --model_name 'text-davinci-003' --input './experiments/q_and_a/prompt.csv' --output './experiments/q_and_a/response.csv' --batch_size 256

# python main.py --exp_name 'feature_and_concept' --model_type 'gpt' --model_name 'text-davinci-003' --input './experiments/feature_and_concept/features.csv' './experiments/feature_and_concept/animals.csv' --output './experiments/feature_and_concept/response.csv' --batch_size 1

# pairwise experiments
# python main.py --exp_name 'pairwise' --model_type 'gpt' --model_name 'text-davinci-003' --input './experiments/pairwise/prompt.csv' --output './experiments/pairwise/response_davinci-003.csv' --batch_size 256
# python main.py --exp_name 'pairwise' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './experiments/pairwise/prompt.csv' --output './experiments/pairwise/response_flan-t5-xl.csv' --batch_size 256
# python main.py --exp_name 'pairwise' --model_type 'flan' --model_name 'google/flan-t5-xxl' --input './experiments/pairwise/prompt.csv' --output './experiments/pairwise/response_flan-t5-xxl.csv' --batch_size 256

# test small group of data
# python main.py --exp_name 'pairwise' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './experiments/pairwise/prompt_small.csv' --output './experiments/pairwise/response_flan-t5-xl.csv' --batch_size 1
# python main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './experiments/triplet/prompt_small.csv' --output './experiments/triplet/response_flan-t5-xl-small.csv' --batch_size 1

# pairwise experiments with 20 times of temp 0.7

for i in $(seq 1 20); do
    python main.py \
        --exp_name 'pairwise' \
        --model_type 'gpt-35-turbo' \
        --input './experiments/pairwise/prompt.csv' \
        --output "./experiments/pairwise/exp_20_temp_0.7/gpt-35-turbo/response_gpt-35-turbo_${i}.csv" \
        --batch_size 256 \
        --temperature 0.7
done

# GPT-3 expriments
# python main.py --exp_name 'pairwise' --model_type 'gpt' --model_name 'text-davinci-002' --input './experiments/pairwise/prompt.csv' --output './experiments/pairwise/response_davinci-002.csv' --batch_size 256
# python main.py --exp_name 'triplet' --model_type 'gpt' --model_name 'text-davinci-002' --input './experiments/triplet/prompt.csv' --output './experiments/triplet/response_temp_default_flipped_davinci-002.csv' --batch_size 256

# Round Things
# python main.py --exp_name 'triplet' --model_type 'gpt' --model_name 'text-davinci-002' \
#     --input './experiments/triplet/roundthings/prompt_roundthings.csv' \
#     --output './experiments/triplet/roundthings/res_rt_temp_default_davinci-002.csv' --batch_size 256

# Flan XXL
# python main.py --exp_name 'pairwise' --model_type 'flan' --model_name 'google/flan-t5-xxl' --input './experiments/pairwise/prompt.csv' --output './experiments/pairwise/flan-t5-xxl_pairwise.csv' --batch_size 256
# python main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xxl' --input './experiments/triplet/prompt.csv' --output './experiments/triplet/flan-t5-xxl_triplet.csv' --batch_size 256

# Llama and Alpaca
# python main.py --exp_name 'pairwise' --model_type 'alpaca' --model_name '7b' --input './experiments/pairwise/prompt.csv' --output './experiments/supplementary/alpaca_pairwise.csv' --batch_size 256
# python main.py --exp_name 'triplet' --model_type 'alpaca' --model_name '7b' --input './experiments/triplet/prompt.csv' --output './experiments/supplementary/alpaca_triplet.csv' --batch_size 256

# python main.py --exp_name 'pairwise' --model_type 'llama' --model_name '7b' --input './experiments/pairwise/prompt.csv' --output './experiments/supplementary/llama_pairwise.csv' --batch_size 256
# python main.py --exp_name 'triplet' --model_type 'llama' --model_name '7b' --input './experiments/triplet/prompt.csv' --output './experiments/supplementary/llama_triplet.csv' --batch_size 256

# Flan ul2
# python main.py --exp_name 'pairwise' --model_type 'flan-ul2' --model_name 'google/flan-ul2' --input './experiments/pairwise/prompt.csv' --output './experiments/supplementary/flan-ul2_pairwise.csv' --batch_size 256
# python main.py --exp_name 'triplet' --model_type 'flan-ul2' --model_name 'google/flan-ul2' --input './experiments/triplet/prompt.csv' --output './experiments/supplementary/flan-ul2_triplet.csv' --batch_size 256

# falcon-7b
# python main.py --exp_name 'pairwise' --model_type 'falcon-7b' --model_name "tiiuae/falcon-7b" \
#     --input './experiments/pairwise/prompt_small.csv' \
#     --output './experiments/supplementary/falcon-7b_pairwise_small.csv'
# python main.py --exp_name 'triplet' --model_type 'falcon-7b' --model_name "tiiuae/falcon-7b" \
#     --input './experiments/triplet/prompt_small.csv' \
#     --output './experiments/supplementary/falcon-7b_triplet_small.csv'

# CoT
# python main.py --exp_name 'triplet' --model_type 'flan-t5-xxl' \
#     --input './experiments/triplet/prompt_small.csv' \
#     --output './experiments/cot/flan-t5-xxl_cot_triplet_small.csv' \
#     --cot True

# python main.py --exp_name 'triplet' --model_type 'gpt-semantics' \
#     --input './experiments/triplet/prompt.csv' \
#     --output './experiments/cot/gpt-semantics_cot_triplet.csv' \
#     --cot True

# python main.py --exp_name 'triplet' --model_type 'text-davinci-003' \
#     --input './experiments/triplet/prompt.csv' \
#     --output './experiments/cot/text-davinci-003_cot_triplet.csv' \
#     --cot True

# python main.py --exp_name 'pairwise' --model_type 'gpt-semantics' \
#     --input './experiments/pairwise/prompt.csv' \
#     --output './experiments/cot/gpt-semantics_cot_pairwise.csv' \
#     --cot True

# python main.py --exp_name 'pairwise' --model_type 'text-davinci-003' \
#     --input './experiments/pairwise/prompt.csv' \
#     --output './experiments/cot/text-davinci-003_cot_pairwise.csv' \
#     --cot True


# for i in $(seq 2 3); do
#     python main.py --exp_name 'triplet' --model_type 'flan-t5-xxl' \
#         --input './experiments/triplet/prompt.csv' \
#         --output "./experiments/cot/flan-t5-xxl_cot_triplet_${i}.csv" \
#         --batch_size 256 --cot True
# done

# for i in $(seq 1 10); do
#     python main.py --exp_name 'pairwise' --model_type 'flan-t5-xxl' \
#         --input './experiments/pairwise/prompt.csv' \
#         --output "./experiments/cot/flan-t5-xxl_cot_pairwise_${i}.csv" \
#         --cot True
# done

# python main.py --exp_name 'triplet' --model_type 'flan-ul2' \
#     --input './experiments/triplet/prompt.csv' \
#     --output './experiments/cot/flan-ul2_cot_triplet.csv' \
#     --cot True

# python main.py --exp_name 'pairwise' --model_type 'flan-ul2' \
#     --input './experiments/pairwise/prompt.csv' \
#     --output './experiments/cot/flan-ul2_cot_pairwise.csv' \
#     --cot True

# python main.py --exp_name 'pairwise' --model_type 'gpt-4' \
#     --input './experiments/pairwise/prompt_small.csv' \
#     --output './experiments/cot/gpt-4_cot_pairwise.csv' \
#     --cot True

# python main.py --exp_name 'pairwise' --model_type 'gpt-4' \
#     --input './experiments/pairwise/prompt_small.csv' \
#     --output './experiments/cot/gpt-4_pairwise.csv' \

# python main.py --exp_name 'triplet' --model_type 'gpt-4' \
#     --input './experiments/triplet/prompt_small.csv' \
#     --output './experiments/cot/gpt-4_cot_triplet.csv' \
#     --cot True

# python main.py --exp_name 'triplet' --model_type 'gpt-4' \
#     --input './experiments/triplet/prompt_small.csv' \
#     --output './experiments/cot/gpt-4_triplet.csv' \