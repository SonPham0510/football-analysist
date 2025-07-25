#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if 'data' directory does not exist and then create it
if [[ ! -e $DIR/data ]]; then
    mkdir "$DIR/data"
else
    echo "'data' directory already exists."
fi

# download the models
gdown -O "$DIR/data/football-ball-detection.pt" "https://drive.google.com/uc?id=1-Z_hEP-LN5U8-b_k6Bnut1WgGXfjbPkt"
gdown -O "$DIR/data/football-player-detection.pt" "https://drive.google.com/uc?id=1c0J0-uD2XkyCNxyBTQU3_vYIkhBVzSjU"
gdown -O "$DIR/data/football-pitch-detection.pt" "https://drive.google.com/uc?id=1lmWnE6rccsC_oZ2RtoA-e3BJW25Jvhq_"

# download the videos
gdown -O "$DIR/data/0bfacc_0.mp4" "https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"
gdown -O "$DIR/data/2e57b9_0.mp4" "https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf"
gdown -O "$DIR/data/08fd33_0.mp4" "https://drive.google.com/uc?id=1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-"
gdown -O "$DIR/data/573e61_0.mp4" "https://drive.google.com/uc?id=1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU"
gdown -O "$DIR/data/121364_0.mp4" "https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"
gdown -O "$DIR/data/test_bra_ger.mp4" "https://drive.google.com/uc?id=10o7wjWJ1w7j-yhoOCu9Au4Soekfau4Oy"
