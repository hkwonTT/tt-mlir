#!/bin/bash

function print_execution() {
    T_LOG_FILE=$1
    cat ${T_LOG_FILE} | grep --color=always -E '(PCC=|Executing operation)' | grep --color=always -vE '(deallocate)|(to_layout)|(from_device)|(to_device)|(get_device)' | sed -E "s/\x1B\[[0-9;]*[a-zA-Z]//g" | sed -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3} - DEBUG - //' |sed -E 's/[[:space:]]*RuntimeTTNN[[:space:]]*\|[[:space:]]*DEBUG[[:space:]]*\|[[:space:]]*Executing operation:[[:space:]]*//g' | sed -E 's/("ttnn\.[a-zA-Z0-9_-]+")/\x1b[33m\1\x1b[0m/g' | sed -E 's/(PCC=[0-9.]+)/\x1b[31m\1\x1b[0m/g'
}
function print_program_level_golden_comparison_result() {
    T_LOG_FILE=$1
    cat ${T_LOG_FILE} | grep --color=always -iE '(program[- ]level)|(test case)|(prgram-level)'

}

T_LOG_FILE=runlog.txt
T_LOG_EXTRACTED_FILE=runlog_extracted.txt
rm -rf ttnn && \
python test_ttir_llama_tile.py && \
{
    ttrt run --save-artifacts --save-golden-tensors ttnn/test_llama_attention_multidevice.ttnn &>  >(tee ${T_LOG_FILE});\
    print_program_level_golden_comparison_result ${T_LOG_FILE}
    print_execution ${T_LOG_FILE} > ${T_LOG_EXTRACTED_FILE};
}
