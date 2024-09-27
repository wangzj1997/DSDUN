####################### Code for Dual-Domain Self-Consistency-Enhanced Deep Unfolding Network for Accelerated MRI Reconstruction ############################

1、train
python models/unfoldingNet/train.py --data-path /ssd/dataset/CC359/Single_channel  --exp-dir checkpoints/radial_kspace_visual/cartesian20/kiki  --use-visdom False

2、test
python models/unfoldingNet/run.py  --data-path /ssd/dataset/CC359/Single_channel --checkpoint 'checkpoints/unfolding_Experience/random10/DFDUN/best_model.pt'  --out-dir reconstructions_val/unfolding_Experience/validation/random10/DFDUN


3、evaluate
python utils_own/evaluate_from_i.py --target-path /ssd/dataset/CC359/Single_channel/Val --predictions-path reconstructions_val/unfolding_Experience/validation/random10/DFDUN --name dfdun_random10_val  --i 0
                                               

4、resume
python models/Unet/train.py --data-path /ssd/dataset/CC359/Single_channel  --use-visdom False  --resume  --checkpoint checkpoints/unet/model.pt


##################### Zero_filled #######################

python models/zero_filled/run_zero_filled.py --data-path /ssd/dataset/CC359/Single_channel/Val --out-path reconstructions_val/zero_filled