#!/bin/bash -l

python3 util/config.py --map Town01
sleep 2
python3 examples/spawn_npc.py -n 50 -w 50 &
python3 examples/dynamic_weather.py --speed 1.5 &

sleep 5

python3 data_collector.py -n 2 -a --res 1280x720 --filter model3 --prefix town_1_multi_modal_scc_1 &
sleep 2
python3 data_collector.py -n 4 -a --res 640x480 --filter model3 --prefix town_1_multi_modal_scc_2 &
sleep 2
python3 data_collector.py -n 6 -a --res 400x300 --filter model3 --prefix town_1_multi_modal_scc_2 &
sleep 2
python3 data_collector.py -n 8 -a --res 400x300 --filter model3 --prefix town_1_multi_modal_scc_3 &
sleep 2
python3 data_collector.py -n 10 -a --res 640x480 --filter model3 --prefix town_1_multi_modal_scc_4 &