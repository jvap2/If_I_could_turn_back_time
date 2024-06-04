#bin/bash

# Run the code

for i in 128 256 512
do
    for j in 10 25 50 100
    do
        for k in 128 256
        do 
            python3 ann_fe.py $i $j $k >> output.csv
        done
    done
done