#!/bin/bash
clear
echo "starting shell script"

gnome-terminal \
--tab --title="Buffer"        -e "bash -c ' ./../../module/buffer/buffer.sh                -i buffer.ini';          bash" \
--tab --title="Playback"        -e "bash -c 'python ../../module/playbacksignal/playbacksignal.py                -i playbacksignal.ini';          bash" \
--tab --title="Preprocessing"        -e "bash -c 'python ../../module/preprocessing/preprocessing.py                -i preprocessing.ini';          bash" \
--tab --title="Spectral_CBC"        -e "bash -c 'python ../../module/spectral_cbc/spectral_cbc.py                -i spectral_cbc.ini';          bash" \
--tab --title="Biotuning"        -e "bash -c 'python ../../module/biotuning/biotuning.py                -i biotuning.ini';          bash" \
--tab --title="Complexity"      -e "bash -c 'python ../../module/complexity/complexity.py            -i complexity.ini';        bash" \
--tab --title="Ratios"        -e "bash -c 'python ../../module/postprocessing/postprocessing.py                -i computeratio.ini';          bash" \
--tab --title="Baseline"      -e "bash -c 'python ../../module/baseline/baseline.py            -i baseline.ini';        bash" \
--tab --title="Smoothing"      -e "bash -c 'python ../../module/historycontrol/historycontrol.py            -i smoothing.ini';        bash" \
--tab --title="OutputOSC"      -e "bash -c 'python ../../module/outputosc/outputosc.py            -i outputosc.ini';        bash" \
--tab --title="Plotsignal"      -e "bash -c 'python ../../module/plotsignal/plotsignal.py            -i plotsignal.ini';        bash" \
--tab --title="Plotspectral"      -e "bash -c 'python ../../module/plotspectral/plotspectral.py            -i plotspectral.ini';        bash" \
