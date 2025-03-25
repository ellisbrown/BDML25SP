#!/bin/bash

# # UPLOAD

# dataset="/scratch/BDML25SP"

# repo_id=ellisbrown/BDML25SP
# HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli upload \
#     $repo_id \
#     $dataset \
#     --repo-type dataset

# download


dataset="$HOME/datasets/BDML25SP"

repo_id=ellisbrown/BDML25SP
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    $repo_id \
    --local-dir $dataset \
    --repo-type dataset