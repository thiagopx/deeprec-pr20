# # default ----------
# python -m exp2_ablation.test --dataset D1 --model-id cdip_0.2_1000_32x32 --results-id D1_0.2_1000_32x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5
# python -m exp2_ablation.test --dataset D2 --model-id cdip_0.2_1000_32x32 --results-id D2_0.2_1000_32x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5
# python -m exp2_ablation.test --dataset cdip --model-id isri-ocr_0.2_1000_32x32 --results-id cdip_0.2_1000_32x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5

# # others ----------

# # neutral_thresh=0.1
# python generate_samples.py --neutral-thresh 0.1 --dataset cdip --max-samples 1000 --sample-size 32 32 --save-dir /mnt/data/samples/deeprec-pr20/cdip_0.1_1000_32x32
# python train.py --model-id cdip_0.1_1000_32x32 --samples-dir /mnt/data/samples/deeprec-pr20/cdip_0.1_1000_32x32
# rm -rf /mnt/data/samples/deeprec-pr20/cdip_0.1_1000_32x32
# python -m exp2_ablation.test --dataset D1 --model-id cdip_0.1_1000_32x32 --results-id D1_0.1_1000_32x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5
# python -m exp2_ablation.test --dataset D2 --model-id cdip_0.1_1000_32x32 --results-id D2_0.1_1000_32x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5
# python generate_samples.py --neutral-thresh 0.1 --dataset isri-ocr --max-samples 1000 --sample-size 32 32 --save-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.1_1000_32x32
# python train.py --model-id isri-ocr_0.1_1000_32x32 --samples-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.1_1000_32x32
# rm -rf /mnt/data/samples/deeprec-pr20/isri-ocr_0.1_1000_32x32
# python -m exp2_ablation.test --dataset cdip --model-id isri-ocr_0.1_1000_32x32 --results-id cdip_0.1_1000_32x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5

# # neutral_thresh=0.2
# python generate_samples.py --neutral-thresh 0.2 --dataset cdip --max-samples 1000 --sample-size 32 32 --save-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_32x32
# python train.py --model-id cdip_0.2_1000_32x32 --samples-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_32x32
# rm -rf /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_32x32
# python generate_samples.py --neutral-thresh 0.2 --dataset isri-ocr --max-samples 1000 --sample-size 32 32 --save-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_32x32
# python train.py --model-id isri-ocr_0.2_1000_32x32 --samples-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_32x32
# rm -rf /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_32x32

# # neutral_thresh=0.3
# python generate_samples.py --neutral-thresh 0.3 --dataset cdip --max-samples 1000 --sample-size 32 32 --save-dir /mnt/data/samples/deeprec-pr20/cdip_0.3_1000_32x32
# python train.py --model-id cdip_0.3_1000_32x32 --samples-dir /mnt/data/samples/deeprec-pr20/cdip_0.3_1000_32x32
# rm -rf /mnt/data/samples/deeprec-pr20/cdip_0.3_1000_32x32
# python -m exp2_ablation.test --dataset D1 --model-id cdip_0.3_1000_32x32 --results-id D1_0.3_1000_32x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5
# python -m exp2_ablation.test --dataset D2 --model-id cdip_0.3_1000_32x32 --results-id D2_0.3_1000_32x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5
# python generate_samples.py --neutral-thresh 0.3 --dataset isri-ocr --max-samples 1000 --sample-size 32 32 --save-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.3_1000_32x32
# python train.py --model-id isri-ocr_0.3_1000_32x32 --samples-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.3_1000_32x32
# rm -rf /mnt/data/samples/deeprec-pr20/isri-ocr_0.3_1000_32x32
# python -m exp2_ablation.test --dataset cdip --model-id isri-ocr_0.3_1000_32x32 --results-id cdip_0.3_1000_32x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5

# # max_samples=500
# python generate_samples.py --neutral-thresh 0.2 --dataset cdip --max-samples 500 --sample-size 32 32 --save-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_500_32x32
# python train.py --model-id cdip_0.2_500_32x32 --samples-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_500_32x32
# rm -rf /mnt/data/samples/deeprec-pr20/cdip_0.2_500_32x32
# python -m exp2_ablation.test --dataset D1 --model-id cdip_0.2_500_32x32 --results-id D1_0.2_500_32x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5
# python -m exp2_ablation.test --dataset D2 --model-id cdip_0.2_500_32x32 --results-id D2_0.2_500_32x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5
# python generate_samples.py --neutral-thresh 0.2 --dataset isri-ocr --max-samples 500 --sample-size 32 32 --save-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_500_32x32
# python train.py --model-id isri-ocr_0.2_500_32x32 --samples-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_500_32x32
# rm -rf /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_500_32x32
# python -m exp2_ablation.test --dataset cdip --model-id isri-ocr_0.2_500_32x32 --results-id cdip_0.2_500_32x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5

# # max_samples=1500
# python generate_samples.py --neutral-thresh 0.2 --dataset cdip --max-samples 1500 --sample-size 32 32 --save-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1500_32x32
# python train.py --model-id cdip_0.2_1500_32x32 --samples-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1500_32x32
# rm -rf /mnt/data/samples/deeprec-pr20/cdip_0.2_1500_32x32
# python -m exp2_ablation.test --dataset D1 --model-id cdip_0.2_1500_32x32 --results-id D1_0.2_1500_32x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5
# python -m exp2_ablation.test --dataset D2 --model-id cdip_0.2_1500_32x32 --results-id D2_0.2_1500_32x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5
# python generate_samples.py --neutral-thresh 0.2 --dataset isri-ocr --max-samples 1500 --sample-size 32 32 --save-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1500_32x32
# python train.py --model-id isri-ocr_0.2_1500_32x32 --samples-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1500_32x32
# rm -rf /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1500_32x32
# python -m exp2_ablation.test --dataset cdip --model-id isri-ocr_0.2_1500_32x32 --results-id cdip_0.2_1500_32x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5

# # sample_size=32x48
# python generate_samples.py --neutral-thresh 0.2 --dataset cdip --max-samples 1000 --sample-size 32 48 --save-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_32x48
# python train.py --model-id cdip_0.2_1000_32x48 --samples-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_32x48
# rm -rf /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_32x48
# python -m exp2_ablation.test --dataset D1 --model-id cdip_0.2_1000_32x48 --results-id D1_0.2_1000_32x48_10 --input-size 3000 48 --vshift 10 --max-ndocs 5
# python -m exp2_ablation.test --dataset D2 --model-id cdip_0.2_1000_32x48 --results-id D2_0.2_1000_32x48_10 --input-size 3000 48 --vshift 10 --max-ndocs 5
# python generate_samples.py --neutral-thresh 0.2 --dataset isri-ocr --max-samples 1000 --sample-size 32 48 --save-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_32x48
# python train.py --model-id isri-ocr_0.2_1000_32x48 --samples-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_32x48
# rm -rf /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_32x48
# python -m exp2_ablation.test --dataset cdip --model-id isri-ocr_0.2_1000_32x48 --results-id cdip_0.2_1000_32x48_10 --input-size 3000 48 --vshift 10 --max-ndocs 5

# # sample_size=32x64
# python generate_samples.py --neutral-thresh 0.2 --dataset cdip --max-samples 1000 --sample-size 32 64 --save-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_32x64
# python train.py --model-id cdip_0.2_1000_32x64 --samples-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_32x64
# rm -rf /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_32x64
# python -m exp2_ablation.test --dataset D1 --model-id cdip_0.2_1000_32x64 --results-id D1_0.2_1000_32x64_10 --input-size 3000 64 --vshift 10 --max-ndocs 5
# python -m exp2_ablation.test --dataset D2 --model-id cdip_0.2_1000_32x64 --results-id D2_0.2_1000_32x64_10 --input-size 3000 64 --vshift 10 --max-ndocs 5
# python generate_samples.py --neutral-thresh 0.2 --dataset isri-ocr --max-samples 1000 --sample-size 32 64 --save-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_32x64
# python train.py --model-id isri-ocr_0.2_1000_32x64 --samples-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_32x64
# rm -rf /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_32x64
# python -m exp2_ablation.test --dataset cdip --model-id isri-ocr_0.2_1000_32x64 --results-id cdip_0.2_1000_32x64_10 --input-size 3000 64 --vshift 10 --max-ndocs 5

# # sample_size=48x32
# python generate_samples.py --neutral-thresh 0.2 --dataset cdip --max-samples 1000 --sample-size 48 32 --save-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_48x32
# python train.py --model-id cdip_0.2_1000_48x32 --samples-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_48x32
# rm -rf /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_48x32
# python -m exp2_ablation.test --dataset D1 --model-id cdip_0.2_1000_48x32 --results-id D1_0.2_1000_48x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5
# python -m exp2_ablation.test --dataset D2 --model-id cdip_0.2_1000_48x32 --results-id D2_0.2_1000_48x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5
# python generate_samples.py --neutral-thresh 0.2 --dataset isri-ocr --max-samples 1000 --sample-size 48 32 --save-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_48x32
# python train.py --model-id isri-ocr_0.2_1000_48x32 --samples-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_48x32
# rm -rf /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_48x32
# python -m exp2_ablation.test --dataset cdip --model-id isri-ocr_0.2_1000_48x32 --results-id cdip_0.2_1000_48x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5

# # sample_size=48x48
# python generate_samples.py --neutral-thresh 0.2 --dataset cdip --max-samples 1000 --sample-size 48 48 --save-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_48x48
# python train.py --model-id cdip_0.2_1000_48x48 --samples-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_48x48
# rm -rf /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_48x48
# python -m exp2_ablation.test --dataset D1 --model-id cdip_0.2_1000_48x48 --results-id D1_0.2_1000_48x48_10 --input-size 3000 48 --vshift 10 --max-ndocs 5
# python -m exp2_ablation.test --dataset D2 --model-id cdip_0.2_1000_48x48 --results-id D2_0.2_1000_48x48_10 --input-size 3000 48 --vshift 10 --max-ndocs 5
# python generate_samples.py --neutral-thresh 0.2 --dataset isri-ocr --max-samples 1000 --sample-size 48 48 --save-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_48x48
# python train.py --model-id isri-ocr_0.2_1000_48x48 --samples-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_48x48
# rm -rf /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_48x48
# python -m exp2_ablation.test --dataset cdip --model-id isri-ocr_0.2_1000_48x48 --results-id cdip_0.2_1000_48x48_10 --input-size 3000 48 --vshift 10 --max-ndocs 5

# # sample_size=48x64
# python generate_samples.py --neutral-thresh 0.2 --dataset cdip --max-samples 1000 --sample-size 48 64 --save-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_48x64
# python train.py --model-id cdip_0.2_1000_48x64 --samples-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_48x64
# rm -rf /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_48x64
# python -m exp2_ablation.test --dataset D1 --model-id cdip_0.2_1000_48x64 --results-id D1_0.2_1000_48x64_10 --input-size 3000 64 --vshift 10 --max-ndocs 5
# python -m exp2_ablation.test --dataset D2 --model-id cdip_0.2_1000_48x64 --results-id D2_0.2_1000_48x64_10 --input-size 3000 64 --vshift 10 --max-ndocs 5
# python generate_samples.py --neutral-thresh 0.2 --dataset isri-ocr --max-samples 1000 --sample-size 48 64 --save-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_48x64
# python train.py --model-id isri-ocr_0.2_1000_48x64 --samples-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_48x64
# rm -rf /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_48x64
# python -m exp2_ablation.test --dataset cdip --model-id isri-ocr_0.2_1000_48x64 --results-id cdip_0.2_1000_48x64_10 --input-size 3000 64 --vshift 10 --max-ndocs 5

# # sample_size=64x32
# python generate_samples.py --neutral-thresh 0.2 --dataset cdip --max-samples 1000 --sample-size 64 32 --save-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_64x32
# python train.py --model-id cdip_0.2_1000_64x32 --samples-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_64x32
# rm -rf /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_64x32
# python -m exp2_ablation.test --dataset D1 --model-id cdip_0.2_1000_64x32 --results-id D1_0.2_1000_64x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5
# python -m exp2_ablation.test --dataset D2 --model-id cdip_0.2_1000_64x32 --results-id D2_0.2_1000_64x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5
# python generate_samples.py --neutral-thresh 0.2 --dataset isri-ocr --max-samples 1000 --sample-size 64 32 --save-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_64x32
# python train.py --model-id isri-ocr_0.2_1000_64x32 --samples-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_64x32
# rm -rf /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_64x32
# python -m exp2_ablation.test --dataset cdip --model-id isri-ocr_0.2_1000_64x32 --results-id cdip_0.2_1000_64x32_10 --input-size 3000 32 --vshift 10 --max-ndocs 5

# # sample_size=64x48
# python generate_samples.py --neutral-thresh 0.2 --dataset cdip --max-samples 1000 --sample-size 64 48 --save-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_64x48
# python train.py --model-id cdip_0.2_1000_64x48 --samples-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_64x48
# rm -rf /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_64x48
# python -m exp2_ablation.test --dataset D1 --model-id cdip_0.2_1000_64x48 --results-id D1_0.2_1000_64x48_10 --input-size 3000 48 --vshift 10 --max-ndocs 5
# python -m exp2_ablation.test --dataset D2 --model-id cdip_0.2_1000_64x48 --results-id D2_0.2_1000_64x48_10 --input-size 3000 48 --vshift 10 --max-ndocs 5
# python generate_samples.py --neutral-thresh 0.2 --dataset isri-ocr --max-samples 1000 --sample-size 64 48 --save-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_64x48
# python train.py --model-id isri-ocr_0.2_1000_64x48 --samples-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_64x48
# rm -rf /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_64x48
# python -m exp2_ablation.test --dataset cdip --model-id isri-ocr_0.2_1000_64x48 --results-id cdip_0.2_1000_64x48_10 --input-size 3000 48 --vshift 10 --max-ndocs 5

# # sample_size=64x64
# python generate_samples.py --neutral-thresh 0.2 --dataset cdip --max-samples 1000 --sample-size 64 64 --save-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_64x64
# python train.py --model-id cdip_0.2_1000_64x64 --samples-dir /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_64x64
# rm -rf /mnt/data/samples/deeprec-pr20/cdip_0.2_1000_64x64
# python -m exp2_ablation.test --dataset D1 --model-id cdip_0.2_1000_64x64 --results-id D1_0.2_1000_64x64_10 --input-size 3000 64 --vshift 10 --max-ndocs 5
# python -m exp2_ablation.test --dataset D2 --model-id cdip_0.2_1000_64x64 --results-id D2_0.2_1000_64x64_10 --input-size 3000 64 --vshift 10 --max-ndocs 5
# python generate_samples.py --neutral-thresh 0.2 --dataset isri-ocr --max-samples 1000 --sample-size 64 64 --save-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_64x64
# python train.py --model-id isri-ocr_0.2_1000_64x64 --samples-dir /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_64x64
# rm -rf /mnt/data/samples/deeprec-pr20/isri-ocr_0.2_1000_64x64
# python -m exp2_ablation.test --dataset cdip --model-id isri-ocr_0.2_1000_64x64 --results-id cdip_0.2_1000_64x64_10 --input-size 3000 64 --vshift 10 --max-ndocs 5

# vshift=0
python -m exp2_ablation.test --dataset D1 --model-id cdip_0.2_1000_32x32 --results-id D1_0.2_1000_32x32_0 --input-size 3000 32 --vshift 0 --max-ndocs 5
python -m exp2_ablation.test --dataset D2 --model-id cdip_0.2_1000_32x32 --results-id D2_0.2_1000_32x32_0 --input-size 3000 32 --vshift 0 --max-ndocs 5
python -m exp2_ablation.test --dataset cdip --model-id isri-ocr_0.2_1000_32x32 --results-id cdip_0.2_1000_32x32_0 --input-size 3000 32 --vshift 0 --max-ndocs 5

# vshift=5
python -m exp2_ablation.test --dataset D1 --model-id cdip_0.2_1000_32x32 --results-id D1_0.2_1000_32x32_5 --input-size 3000 32 --vshift 5 --max-ndocs 5
python -m exp2_ablation.test --dataset D2 --model-id cdip_0.2_1000_32x32 --results-id D2_0.2_1000_32x32_5 --input-size 3000 32 --vshift 5 --max-ndocs 5
python -m exp2_ablation.test --dataset cdip --model-id isri-ocr_0.2_1000_32x32 --results-id cdip_0.2_1000_32x32_5 --input-size 3000 32 --vshift 5 --max-ndocs 5

# vshift=15
python -m exp2_ablation.test --dataset D1 --model-id cdip_0.2_1000_32x32 --results-id D1_0.2_1000_32x32_15 --input-size 3000 32 --vshift 15 --max-ndocs 5
python -m exp2_ablation.test --dataset D2 --model-id cdip_0.2_1000_32x32 --results-id D2_0.2_1000_32x32_15 --input-size 3000 32 --vshift 15 --max-ndocs 5
python -m exp2_ablation.test --dataset cdip --model-id isri-ocr_0.2_1000_32x32 --results-id cdip_0.2_1000_32x32_15 --input-size 3000 32 --vshift 15 --max-ndocs 5

# vshift=20
python -m exp2_ablation.test --dataset D1 --model-id cdip_0.2_1000_32x32 --results-id D1_0.2_1000_32x32_20 --input-size 3000 32 --vshift 20 --max-ndocs 5
python -m exp2_ablation.test --dataset D2 --model-id cdip_0.2_1000_32x32 --results-id D2_0.2_1000_32x32_20 --input-size 3000 32 --vshift 20 --max-ndocs 5
python -m exp2_ablation.test --dataset cdip --model-id isri-ocr_0.2_1000_32x32 --results-id cdip_0.2_1000_32x32_20 --input-size 3000 32 --vshift 20 --max-ndocs 5