type='tblstm'
conf_thresh=0.01
nms_thresh=0.45
top_k=200
set_file_name='val'
detection='yes'
gpu_id='1'
attention='yes'
python ../eval.py \
    --model_dir '/home/sean/Documents/ssd.pytorch/weights/VIDtssd/trained_model/ssd300_VID2017_6543.pth' \
    --model_name 'ssd300' \
    --literation 15000 \
    --save_folder '../eval' \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --set_file_name $set_file_name \
    --dataset_name 'VID2017' \
    --tssd $type \
    --gpu_id $gpu_id \
    --detection $detection \
    --attention $attention \
    --tub 0 \
    --tub_thresh 0.95 \
    --tub_generate_score 0.5

