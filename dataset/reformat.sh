export PATH=$PATH:/usr/local/ffmpeg/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/ffmpeg/lib

python reformat.py --start_batch $1 --end_batch $2 --gpu_id $3