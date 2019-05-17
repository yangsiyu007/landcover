while getopts ":t:g:" opt; do
    case ${opt} in
        g )
            CUDA_VISIBLE_DEVICES=$OPTARG
            ;;
        t )
            TEST_REGION=$OPTARG
            ;;
        \? )
            echo "Invalid option: $OPTARG" 1>&2
            ;;
        : )
            echo "Invalid option: $OPTARG requires an argument" 1>&2
            ;;
    esac
done
shift $((OPTIND -1))

export PYTHONPATH=.


echo "model, area, tile_type, tile_index, mean_IoU, pixel_accuracy, tile_path, predictions_path"

NUMS_PATCHES=(10 40 100 200 400 1000 2000)
#NUMS_PATCHES=(1000 2000)


# Test original model
#python training/pytorch/test_finetuning.py --area test${TEST_REGION} --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/baseline_models/baseline_$#python training/pytorch/test_finetuning.py --area test${TEST_REGION} --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/baseline_models/baseline_$

for num_patches in ${NUMS_PATCHES[*]}
do
    MODELS_DIR="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/${num_patches}_patches_one_point"
#
    for random_seed in {1..5}
#
    do
#       # Train fine-tuned model
        python training/pytorch/model_finetuning.py \
               --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/checkpoint_best.pth.tar" \
               --training_patches_fn "training/data/finetuning/sampled/test${TEST_REGION}_train_patches_rand_${num_patches}_${random_seed}.txt" \
               --validation_patches_fn "training/data/finetuning/sampled/test${TEST_REGION}_train_patches_rand_${num_patches}_${random_seed}.txt" \
               --log_fn "${MODELS_DIR}/rand_${num_patches}_${random_seed}/train_results.csv" \
               --model_output_directory "${MODELS_DIR}/rand_${num_patches}_${random_seed}"

        # Test fine-tuned models
        MODELS=(
        #    "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/${num_patches}_patches_one_point/rand_${num_patches}_${random_seed}/fin$            "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/${num_patches}_patches_one_point/rand_${num_patches}_${random_seed}/fine$         #   "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/${num_patches}_patches_one_point/rand_${num_patches}_${random_seed}/fin$          #  "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/${num_patches}_patches_one_point/rand_${num_patches}_${random_seed}/fin$
        #    "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/${num_patches}_patches_one_point/rand_${num_patches}_${random_seed}/fin$
 #           "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/${num_patches}_patches_one_point/rand_${num_patches}_${random_seed}/fine$
         #   "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/${num_patches}_patches_one_point/rand_${num_patches}_${random_seed}/fin$
          #  "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/${num_patches}_patches_one_point/rand_${num_patches}_${random_seed}/fin$
        )

  #      for model_file in ${MODELS[*]}
 #       do
#           python training/pytorch/test_finetuning.py --area test${TEST_REGION} --model_file "$model_file" --test_tile_fn training/data/finetuning/test${TEST_REGION}_test_tiles.txt --tile_type te$            python training/pytorch/test_finetuning.py --area test${TEST_REGION} --model_file "$model_file" --test_tile_fn training/data/finetuning/test${TEST_REGION}_train_tiles.txt --tile_type t$        done
#    done
done
done

