#!/bin/bash

usage(){
  echo "Usage: $0 <undLine> <PVprefix> <serialNumber> <camNumber>"
  echo "eg.:   $0 {sxr|hxr} INTRSP:B081:CAL0: SNxxx 1"
  exit 1
}

# invoke  usage
# call usage() function if filename not supplied
[[ $# -eq 0 ]] && usage

LINE=$1
SEGMENT=$2
SERIAL=$3
CAM=$4

python calibrate_cams.py --line ${LINE} --calibrate ${SEGMENT} --serial ${SERIAL} -n ${CAM} --velocity 1.4 --dwell 0.2 --store-to-pv
