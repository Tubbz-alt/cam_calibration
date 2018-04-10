#!/bin/bash

input_file=$1
output_file=${1}_newformat.txt

cat $input_file | sed \
    -e s/^.*CALGDRPOT/LINEAR_POT_/ \
    -e s/^.*CALCAMPOT/ROTARY/ \
    -e s/^.*CALCAMANGLE/ANGLES/ \
    -e s/^.*CALVOLTAVG/AVERAGE_INPUT_VOLTAGE/ \
    -e s/^gain_rmsFit/gain_rms_fit/ \
    -e s/^linear_offset/linear_phase_offset/ \
    -e s/^rotaryPotOffset/rotary_pot_offset/ \
    > $output_file

echo "prefix = testing:" >> $output_file
