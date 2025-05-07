CUDA_VISIBLE_DEVICES=1 python protomotions/train_agent.py +exp=full_body_tracker/mlp_single_motion_amp_flat_terrain +robot=t1 +simulator=isaacgym motion_file=/var/local/zifan/motion_data_x/CMU-t1_retargeted/01/01_01_stageii.npy +experiment_name=mlp_full_body_tracker_t1_0401 num_envs=4096 +opt=wandb ngpu=1
```

Getup motions
```
[
    '0-CMU_74_74_08_poses', 
    '0-KIT_442_Aufstehen01_poses', 
    '0-KIT_3_kneel_up_with_right_hand01_poses', 
    '0-CMU_111_111_08_poses', 
    '0-CMU_139_139_18_poses', 
    '0-CMU_77_77_18_poses', 
    '0-ACCAD_Female1General_c3d_A10 - lie to crouch_poses', 
    '0-KIT_3_kneel_up_from_crawl01_poses', 
    '0-CMU_77_77_17_poses', 
    '0-CMU_111_111_07_poses', 
    '0-KIT_3_kneel_up_with_left_hand01_poses', 
    '0-CMU_77_77_16_poses', 
    '0-CMU_84_84_08_poses', 
    '0-CMU_139_139_16_poses', 
    '0-CMU_140_140_02_poses', 
    '0-CMU_22_23_Rory_22_19_poses', 
    '0-CMU_22_23_justin_22_20_poses', 
    '0-CMU_139_139_17_poses', 
    '0-CMU_114_114_02_poses', 
    '0-CMU_140_140_01_poses', 
    '0-ACCAD_Male2MartialArtsStances_c3d_D13 -crouch to ready_poses', 
    '0-CMU_140_140_09_poses', 
    '0-CMU_22_23_justin_22_18_poses', 
    '0-CMU_140_140_04_poses', 
    '0-CMU_140_140_08_poses']
```

Motion retarget
```
python data/scripts/convert_amass_to_isaac.py /var/local/zifan/motion_data_x --robot-type=t1 --humanoid-type smplx --force-retarget
```

Steering task
```
CUDA_VISIBLE_DEVICES=1 python protomotions/train_agent.py +exp=steering_amp_mlp +robot=t1 +simulator=isaacgym +experiment_name=t1_steering_amp +motion_file=/var/local/zifan/motion_data_x/CMU-t1_retargeted/02/02_01_stageii.npy
```

Full body tracking T1
```
CUDA_VISIBLE_DEVICES=1 python protomotions/train_agent.py +exp=full_body_tracker/transformer_flat_terrain +robot=t1 +simulator=isaacgym motion_file=/var/local/zifan/motion_data_x/amass_t1_train.yaml +experiment_name=t1_full_body_tracker
```


export LD_LIBRARY_PATH=~/miniconda3/envs/protomotions/lib/:$LD_LIBRARY_PATH


python protomotions/eval_agent.py \
    +checkpoint=results/mlp_full_body_tracker_h1_0422/last.ckpt \
    +headless=False \
    +robot=h1 \
    +simulator=isaacgym \
    +motion_file=data/yaml_files/hml3d_h1_2.yaml \
    +num_envs=1


python data/scripts/convert_amass_to_isaac.py /home/linji/amass/ --robot-type=h1 --humanoid-type smplx --force-retarget

python data/scripts/process_hml3d_data.py hml3d_h1_3.yaml amass/ --occlusion-data-path=data/amass/amassx_occlusion_v1.pkl --humanoid-type=smplx --motion-fps-path=data/yaml_files/motion_fps_smplx_h1_3.yaml 


python protomotions/train_agent.py  +exp=full_body_tracker/transformer_flat_terrain +robot=h1 +simulator=isaacgym +motion_file=/home/linji/nfs/ProtoMotions-T1/data/yaml_files/hml3d_h1_3.yaml +experiment_name=full_body_tracker_h1_cmudata ++num_envs=256 +opt=wandb

HYDRA_FULL_ERROR=1 python protomotions/eval_agent.py +robot=t1 +simulator=isaacgym +checkpoint=results/t1_full_body_tracker/last.ckpt +motion_file=/home/linji/nfs/ProtoMotions-T1/motion_data_x/CMU-t1_retargeted/140/140_01_stageii.npy

HYDRA_FULL_ERROR=1 python protomotions/eval_agent.py +robot=t1 +simulator=isaacgym +checkpoint=results/t1_full_body_tracker_amp/last.ckpt +motion_file=/home/linji/nfs/ProtoMotions-T1/motion_data_x/CMU-t1_retargeted/140/140_01_stageii.npy

HYDRA_FULL_ERROR=1 python protomotions/train_agent.py +exp=full_body_tracker/transformer_flat_terrain +robot=t1 +simulator=isaacgym motion_file=motion_data_x/amass_t1_train.yaml +experiment_name=t1_full_body_tracker

HYDRA_FULL_ERROR=1 python protomotions/train_agent.py +exp=full_body_tracker/transformer_flat_terrain +robot=t1 +simulator=isaacgym motion_file=/var/local/zifan/motion_data_x/amass_t1_train.yaml +experiment_name=t1_full_body_tracker

python protomotions/train_agent.py  +exp=full_body_tracker/transformer_flat_terrain +robot=t1 +simulator=isaacgym +motion_file=/home/linji/nfs/ProtoMotions-T1/motion_data_x/amass_t1_train.yaml +experiment_name=full_body_tracker_t1_cmudata +opt=wandb

HYDRA_FULL_ERROR=1 python protomotions/eval_agent.py +robot=t1 +simulator=isaacgym +checkpoint=results/t1_full_body_tracker_amp/last.ckpt +motion_file=/home/linji/nfs/ProtoMotions-T1/motion_data_x/amass_t1_train.yaml ++num_envs=10

python protomotions/train_agent.py  +exp=full_body_tracker/transformer_flat_terrain +robot=t1 +simulator=isaacgym +motion_file=/home/linji/nfs/ProtoMotions-T1/motion_data_x/amass_t1_train.yaml +experiment_name=full_body_tracker_t1_cmudata_02 +opt=wandb ++num_envs=4096

export LD_LIBRARY_PATH=~/miniconda3/envs/protomotions/lib/:$LD_LIBRARY_PATH


python protomotions/train_agent.py \
    +exp=full_body_tracker/transformer_flat_terrain \
    +robot=t1 \
    +simulator=isaacgym \
    motion_file=/home/linji/nfs/ProtoMotions-T1/motion_data_x/amass_t1_train.yaml \
    +experiment_name=t1_full_body_tracker_plr_v1 \
    +opt=wandb
    # Ensure the loaded agent config has plr.enabled=true


python protomotions/train_agent.py     +exp=full_body_tracker/transformer_flat_terrain     +robot=t1     +simulator=isaacgym     motion_file=/home/linji/nfs/ProtoMotions-T1/motion_data_x/amass_t1_train.yaml     +experiment_name=t1_full_body_tracker_plr_v1     +opt=wandb


HYDRA_FULL_ERROR=1 python protomotions/train_agent.py +exp=masked_mimic/flat_terrain +robot=t1 +simulator=isaacgym motion_file=/home/linji/nfs/ProtoMotions-T1/motion_data_x/amass_t1_train.yaml agent.config.expert_model_path=/home/linji/nfs/ProtoMotions-T1/results/t1_full_body_tracker_plr_v2/ +experiment_name=t1_masked_mimic_plr_v2 +opt=wandb

HYDRA_FULL_ERROR=1 python protomotions/train_agent.py +exp=masked_mimic/flat_terrain +robot=t1 +simulator=isaacgym motion_file=/home/linji/nfs/ProtoMotions-T1/motion_data_x/amass_t1_train.yaml agent.config.expert_model_path=/home/linji/nfs/ProtoMotions-T1/results/t1_full_body_tracker_plr_v2/ +experiment_name=t1_masked_mimic_plr_v2 +opt=wandb ++num_envs=256

python protomotions/eval_agent.py +robot=t1 +simulator=isaacgym +checkpoint=results/t1_masked_mimic_plr_v2/last.ckpt +motion_file=/home/linji/nfs/ProtoMotions-T1/motion_data_x/amass_t1_train.yaml +opt=[masked_mimic/tasks/user_control]

(protomotions) linji@robotixx-c2-v101:~/nfs/ProtoMotions-T1$ python protomotions/eval_agent.py +robot=t1 +simulator=isaacgym +checkpoint=results/t1_masked_mimic_plr_v2/last.ckpt +motion_file=/home/linji/nfs/ProtoMotions-T1/motion_data_x/amass_t1_train.yaml

# Evaluate MaskedMimic (Stage 2) for Text-to-Motion
python protomotions/eval_agent.py \
    +robot=t1 \
    +simulator=isaacgym \
    +checkpoint=results/t1_masked_mimic_plr_v2/last.ckpt \
    +motion_file=/home/linji/nfs/ProtoMotions-T1/motion_data_x/amass_t1_train.yaml \
    +opt=[masked_mimic/tasks/text_control] \
    ++num_envs=1 \
    headless=False

python protomotions/eval_agent.py \
    +robot=t1 \
    +simulator=isaacgym \
    +checkpoint=results/t1_masked_mimic_plr_v2/last.ckpt \
    +motion_file=/home/linji/nfs/ProtoMotions-T1/motion_data_x/amass_t1_train.yaml \
    +opt=[masked_mimic/constraints/hands]



python protomotions/eval_agent.py \
    +robot=t1 \
    +simulator=isaacgym \
    +checkpoint=results/t1_masked_mimic_plr_v2/last.ckpt \
    +motion_file=/home/linji/nfs/ProtoMotions-T1/motion_data_x/amass_t1_train.yaml \
    +opt=[masked_mimic/masked_mimic_text_control_task]


HYDRA_FULL_ERROR=1 python protomotions/train_agent.py +exp=masked_mimic/flat_terrain +robot=t1 +simulator=isaacgym motion_file=/home/linji/nfs/ProtoMotions-T1/motion_data_x/amass_t1_train.yaml agent.config.expert_model_path=/home/linji/nfs/ProtoMotions-T1/results/t1_full_body_tracker_plr_v2/ +experiment_name=t1_masked_mimic_plr_v2_debug_text +opt=wandb ++num_envs=256 agent.config.max_epochs=1


HYDRA_FULL_ERROR=1 python protomotions/eval_agent.py     +robot=t1     +simulator=isaacgym     +checkpoint=results/t1_masked_mimic_plr_v2/last.ckpt     +motion_file=/home/linji/nfs/ProtoMotions-T1/motion_data_x/amass_t1_train.yaml     +opt=[masked_mimic/masked_mimic_text_control_task]