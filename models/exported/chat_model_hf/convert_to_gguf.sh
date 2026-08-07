#!/bin/bash
# Convert chat_model to GGUF
python "G:/PROJECTS/MyIAModelChat/models/exported/convert.py" "G:/PROJECTS/MyIAModelChat/models/exported/chat_model_hf" --outfile "G:/PROJECTS/MyIAModelChat/models/exported/chat_model_q8_k.gguf" --outtype q8_k
