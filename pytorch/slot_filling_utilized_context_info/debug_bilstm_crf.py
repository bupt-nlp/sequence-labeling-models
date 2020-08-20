import subprocess

command = 'rm -rf ./output'
subprocess.call(command.split())

import json
import shutil
import sys

from allennlp.commands import main

## 对Allennlp代码进行调试

# allennlp train -s=./output/bilstm_crf -r ./bilstm_fc.json

config_file = "bilstm_crf.json"

# Use overrides to train on CPU.
overrides = json.dumps({"trainer": {"cuda_device": 2}})

serialization_dir = "./output/bilstm_crf"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "--include-package", "packages",
    "-s", serialization_dir,
    "-o", overrides,
]

main()