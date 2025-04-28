#!/bin/bash

function print_execution() {
    cat runlog.txt | grep --color=always -E 'Executing operation' | grep --color=always -vE '(deallocate)|(to_layout)|(from_device)|(to_device)|(get_device)' | sed -E "s/\x1B\[[0-9;]*[a-zA-Z]//g" | sed -E 's/[[:space:]]*RuntimeTTNN[[:space:]]*\|[[:space:]]*DEBUG[[:space:]]*\|[[:space:]]*Executing operation:[[:space:]]*//g' | sed -E 's/("ttnn\.[a-zA-Z0-9_-]+")/\x1b[33m\1\x1b[0m/g'
}
rm -rf ttnn && \
python test_ttir_llama_tile.py && \
ttrt run ttnn/test_llama_attention_multidevice.ttnn &>  >(tee runlog.txt) && \
cat runlog.txt | grep --color=always -iE '(program[- ]level)|(test case)'
# print_execution
