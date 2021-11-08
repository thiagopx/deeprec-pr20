from itertools import product
from .config import CONFIG_SAMPLES_DIR


# samples generation
txt = '''
python generate_samples.py --neutral-thresh 0.2 --dataset cdip --max-samples 1000 --sample-size 32 32 --save-dir {}/cdip_0.2_1000_32x32
python generate_samples.py --neutral-thresh 0.2 --dataset isri-ocr --max-samples 1000 --sample-size 32 32 --save-dir {}/isri-ocr_0.2_1000_32x32
'''.format(*(2 * [CONFIG_SAMPLES_DIR])
)

# training
txt += '''
python train.py --model-id cdip_0.2_1000_32x32 --samples-dir {}/cdip_0.2_1000_32x32
python train.py --model-id isri_ocr_0.2_1000_32x32 --samples-dir {}/isri-ocr_0.2_1000_32x32
'''.format(*(2 * [CONFIG_SAMPLES_DIR]))

# global (big) matrix generation
txt += '''
python -m exp1_proposed.gen_matrix --dataset D1 --model-id cdip_0.2_1000_32x32
python -m exp1_proposed.gen_matrix --dataset D2 --model-id cdip_0.2_1000_32x32
python -m exp1_proposed.gen_matrix --dataset cdip --model-id isri-ocr_0.2_1000_32x32
'''

# test (using the global matrices)
txt += '''
python -m exp1_proposed.test --dataset D1 --nproc 100
python -m exp1_proposed.test --dataset D2 --nproc 100
python -m exp1_proposed.test --dataset cdip --nproc 100
'''

open('exp1_proposed/run.sh', 'w').write(txt)