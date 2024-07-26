#!/bin/bash

for Vd in {1,2,3,4,5,6}; do
    python3 27site_mote2_withreadFile.py 3.0 $Vd > "out_27sites_Nh=18_ep=10_theta=3.0_Vd=$Vd.txt" 2>&1 &
    wait $!
done
