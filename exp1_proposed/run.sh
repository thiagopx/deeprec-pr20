
python generate_samples.py --neutral-thresh 0.2 --dataset cdip --max-samples 1000 --sample-size 32 32 --save-dir /home/tpaixao/samples/deeprec-pr20/cdip_0.2_1000_32x32
python generate_samples.py --neutral-thresh 0.2 --dataset isri-ocr --max-samples 1000 --sample-size 32 32 --save-dir /home/tpaixao/samples/deeprec-pr20/isri-ocr_0.2_1000_32x32

python train.py --model-id cdip_0.2_1000_32x32 --samples-dir /home/tpaixao/samples/deeprec-pr20/cdip_0.2_1000_32x32
python train.py --model-id isri_ocr_0.2_1000_32x32 --samples-dir /home/tpaixao/samples/deeprec-pr20/isri-ocr_0.2_1000_32x32

python -m exp1_proposed.gen_matrix --dataset D1 --model-id cdip_0.2_1000_32x32
python -m exp1_proposed.gen_matrix --dataset D2 --model-id cdip_0.2_1000_32x32
python -m exp1_proposed.gen_matrix --dataset cdip --model-id isri-ocr_0.2_1000_32x32

python -m exp1_proposed.test --dataset D1 --nproc 100
python -m exp1_proposed.test --dataset D2 --nproc 100
python -m exp1_proposed.test --dataset cdip --nproc 100