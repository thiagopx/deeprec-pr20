python -m comparison.proposed_matrix --dataset D1 --model-id cdip_0.2_1000_32x32 --results-id D1_matrix
python -m comparison.proposed_matrix --dataset D2 --model-id cdip_0.2_1000_32x32 --results-id D2_matrix
python -m comparison.proposed_matrix --dataset cdip --model-id isri-ocr_0.2_1000_32x32 --results-id cdip_matrix
python -m comparison.proposed --dataset D1 --nproc 100 --solver conc
python -m comparison.proposed --dataset D2 --nproc 100 --solver conc
python -m comparison.proposed --dataset cdip --nproc 100 --solver conc
python -m comparison.proposed --dataset D1 --nproc 100 --solver nn
python -m comparison.proposed --dataset D2 --nproc 100 --solver nn
python -m comparison.proposed --dataset cdip --nproc 100 --solver nn

python -m comparison.marques_matrix --dataset D1
python -m comparison.marques_matrix --dataset D2
python -m comparison.marques_matrix --dataset cdip
python -m comparison.marques --dataset D1 --nproc 100
python -m comparison.marques --dataset D2 --nproc 100
python -m comparison.marques --dataset cdip --nproc 100

python -m comparison.paixao --dataset D1
python -m comparison.paixao --dataset D2
python -m comparison.paixao --dataset cdip

export OMP_NUM_THREADS=48
python -m comparison.liang --dataset D1 --soft-path /home/thiagopx/docreassembly --num-threads 48
python -m comparison.liang --dataset D2 --soft-path /home/thiagopx/docreassembly --num-threads 48

export OMP_NUM_THREADS=12
python -m comparison.liang --dataset D1 --soft-path /home/thiagopx/docreassembly --num-threads 12
python -m comparison.liang --dataset D2 --soft-path /home/thiagopx/docreassembly --num-threads 12

export OMP_NUM_THREADS=1
python -m comparison.liang --dataset D1 --soft-path /home/thiagopx/docreassembly --num-threads 1
python -m comparison.liang --dataset D2 --soft-path /home/thiagopx/docreassembly --num-threads 1
#python -m comparison.liang --dataset cdip