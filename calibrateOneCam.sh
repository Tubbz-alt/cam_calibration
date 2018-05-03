#!/bin/bash

usage(){
  echo "Usage: $0  <serialNumber> <camNumber>"
  echo "eg.:   $0  SNxxx 1"
  exit 1
}

# invoke  usage
# call usage() function if filename not supplied
[[ $# -eq 0 ]] && usage

SERIAL=$1
CAM=$2

python calibrate_cams.py --line sxr --calibrate INTRSP:B081:CAL0: --serial ${SERIAL} -n ${CAM} --velocity 1.4 --dwell 0.2 --store-to-pv
