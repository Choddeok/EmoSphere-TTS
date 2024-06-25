DEVICE=0
CONFIG="egs/egs_bases/tts/fs2_orig.yaml"
binary_data_dir="/workspace/choddeok/hd0/dataset/binary256/ESD_VAD_rescale_polar_Neu_IQR_spk_cpu_Neu_allx"

run_infer() {
    local model_name=$1
    local task_class=$2
    local hparams="binary_data_dir=$binary_data_dir,task_cls=$task_class"

    # Train
    CUDA_VISIBLE_DEVICES=$DEVICE python run.py --config $CONFIG --exp_name $model_name --reset --hparams="$hparams"

    # Infer with different configurations
    for config_suffix in "" _0.1 _0.5 _0.9; do
        CUDA_VISIBLE_DEVICES=$DEVICE python run.py --config "egs/datasets/audio/esd${config_suffix}/fs_orig.yaml" --exp_name $model_name --reset --infer --hparams="$hparams"
    done

    for config_suffix in "" _0.1 _0.5 _0.9; do
        CUDA_VISIBLE_DEVICES=$DEVICE python run.py --config "egs/datasets/audio/esd${config_suffix}_I/fs_orig.yaml" --exp_name $model_name --reset --infer --hparams="$hparams"
    done
    
    for config_suffix in "" _0.1 _0.5 _0.9; do
        CUDA_VISIBLE_DEVICES=$DEVICE python run.py --config "egs/datasets/audio/esd${config_suffix}_III/fs_orig.yaml" --exp_name $model_name --reset --infer --hparams="$hparams"
    done
    
    for config_suffix in "" _0.1 _0.5 _0.9; do
        CUDA_VISIBLE_DEVICES=$DEVICE python run.py --config "egs/datasets/audio/esd${config_suffix}_V/fs_orig.yaml" --exp_name $model_name --reset --infer --hparams="$hparams"
    done

    for config_suffix in "" _0.1 _0.5 _0.9; do
        CUDA_VISIBLE_DEVICES=$DEVICE python run.py --config "egs/datasets/audio/esd${config_suffix}_VII/fs_orig.yaml" --exp_name $model_name --reset --infer --hparams="$hparams"
    done
}

#########################
#   Run for the model   #
#########################
run_infer "240429_EmoSphere" "tasks.tts.EmoSphere.FastSpeech2OrigTask"