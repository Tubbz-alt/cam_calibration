#!/bin/bash

usage(){
  echo "Usage: $0  <undLine>  <PVprefix>         <serialNumber>"
  echo "eg.:   $0  {sxr|hxr}  INTRSP:B081:LTTS0  LTTS0"
  exit 1
}

# invoke  usage
# call usage() function if filename not supplied
[[ $# -eq 0 ]] && usage

LINE=$1
SEGMENT=$2
SERIAL=$3

for cam in {1..5}; do 
  python calibrate_cams.py --line ${LINE} --calibrate ${SEGMENT}: --serial ${SERIAL} -n ${cam} --velocity 1.4 --dwell 0.2 --store-to-pv
done 

