set -ex
#character_id=0118_cyz_stand
character_id=$1
if [ -z "$character_id" ];then
  echo "Usage: script.sh character_id"
  exit
fi
echo Processing ${character_id} !!!!!!!!
source activate dh
cd data_utils
#python process.py ../dataset/${character_id}/1080p.mp4 --asr hubert
#python process.py ../dataset/0903_mzw/1080p_train_raw_300_500.mp4 --asr hubert
cd ..
#python syncnet.py --save_dir syncnet_ckpts/${character_id} --dataset_dir dataset/${character_id} --asr hubert
#python syncnet.py --save_dir syncnet_ckpts/0903_mzw --dataset_dir dataset/0903_mzw --asr hubert

#python syncnet.py --save_dir syncnet_ckpts/reta_1106 --dataset_dir dataset/reta_1106 --asr hubert
#python syncnet.py --save_dir syncnet_ckpts/zyz_1106 --dataset_dir dataset/zyz_1106 --asr hubert

python train.py --dataset_dir dataset/${character_id}/ --save_dir  checkpoints/${character_id} --asr hubert --use_syncnet --syncnet_checkpoint syncnet_ckpts/${character_id}/50.pth
#python train.py --dataset_dir dataset/0903_mzw/ --save_dir  checkpoints/0903_mzw --asr hubert --use_syncnet --syncnet_checkpoint syncnet_ckpts/0903_mzw/50.pth

#python train.py --dataset_dir dataset/reta_1106/ --save_dir  checkpoints/reta_1106 --asr hubert --use_syncnet --syncnet_checkpoint syncnet_ckpts/reta_1106/39.pth
#python train.py --dataset_dir dataset/zyz_1106/ --save_dir  checkpoints/zyz_1106 --asr hubert --use_syncnet --syncnet_checkpoint syncnet_ckpts/zyz_1106/39.pth


#python inference.py --asr hubert --dataset dataset/huizhang_1106/ --audio_feat demo/female_demo_20s_hu.npy  --save_path results/temp.mp4 --checkpoint checkpoints/huizhang_1106/200.pth