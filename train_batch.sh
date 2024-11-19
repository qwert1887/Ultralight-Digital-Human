set -ex
source activate dh
cd data_utils
#python process.py ../dataset/0913_wang/650p_5min_beauty.mp4 --asr hubert
#python process.py ../dataset/0903_mzw/1080p_train_raw_300_500.mp4 --asr hubert
cd ..
#python syncnet.py --save_dir syncnet_ckpts/0913_wang --dataset_dir dataset/0913_wang --asr hubert
#python syncnet.py --save_dir syncnet_ckpts/0903_mzw --dataset_dir dataset/0903_mzw --asr hubert

#python syncnet.py --save_dir syncnet_ckpts/reta_1106 --dataset_dir dataset/reta_1106 --asr hubert
#python syncnet.py --save_dir syncnet_ckpts/zyz_1106 --dataset_dir dataset/zyz_1106 --asr hubert

python train.py --dataset_dir dataset/0913_wang/ --save_dir  checkpoints/0913_wang --asr hubert --use_syncnet --syncnet_checkpoint syncnet_ckpts/0913_wang/50.pth
python train.py --dataset_dir dataset/0903_mzw/ --save_dir  checkpoints/0903_mzw --asr hubert --use_syncnet --syncnet_checkpoint syncnet_ckpts/0903_mzw/50.pth

#python train.py --dataset_dir dataset/reta_1106/ --save_dir  checkpoints/reta_1106 --asr hubert --use_syncnet --syncnet_checkpoint syncnet_ckpts/reta_1106/39.pth
#python train.py --dataset_dir dataset/zyz_1106/ --save_dir  checkpoints/zyz_1106 --asr hubert --use_syncnet --syncnet_checkpoint syncnet_ckpts/zyz_1106/39.pth


#python inference.py --asr hubert --dataset dataset/huizhang_1106/ --audio_feat demo/female_demo_20s_hu.npy  --save_path results/temp.mp4 --checkpoint checkpoints/huizhang_1106/200.pth