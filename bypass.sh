#python train.py --model-id cdip_0.2_1000_32x32_sn-bypass --arch sn-bypass --num-epochs 30 --samples-dir ~/samples_v3/cdip_0.2_1000_32x32
#python train.py --model-id isri-ocr_0.2_1000_32x32_sn-bypass --arch sn-bypass --num-epochs 30 --samples-dir ~/samples_v3/isri-ocr_0.2_1000_32x32

# cross-database
LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64 python -m ablation.run --dataset D1 --model-id cdip_0.2_1000_32x32_sn-bypass --results-id D1_0.2_1000_32x32_10_sn-bypass --arch sn-bypass --input-size 3000 32 --vshift 10 --max-ndocs 5
LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64 python -m ablation.run --dataset D2 --model-id cdip_0.2_1000_32x32_sn-bypass --results-id D2_0.2_1000_32x32_10_sn-bypass --arch sn-bypass --input-size 3000 32 --vshift 10 --max-ndocs 5
LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64 python -m ablation.run --dataset cdip --model-id isri-ocr_0.2_1000_32x32_sn-bypass --results-id cdip_0.2_1000_32x32_10_sn-bypass --arch sn-bypass --input-size 3000 32 --vshift 10 --max-ndocs 5

# non cross-database
LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64 python -m ablation.run --dataset D1 --model-id isri-ocr_0.2_1000_32x32_sn-bypass --results-id D1_0.2_1000_32x32_10_sn-bypass_isri-ocr --arch sn-bypass --input-size 3000 32 --vshift 10 --max-ndocs 5
LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64 python -m ablation.run --dataset D2 --model-id isri-ocr_0.2_1000_32x32_sn-bypass --results-id D2_0.2_1000_32x32_10_sn-bypass_isri-ocr --arch sn-bypass --input-size 3000 32 --vshift 10 --max-ndocs 5