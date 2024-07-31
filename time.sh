backends=('eager' 'hidet' 'inductor')
PY_EXE=/home/caohanghang/miniconda3/envs/torch230/bin/python
PROG=/home/caohanghang/gitproject/QTO/main.py
PROJ_ROOT=/home/caohanghang/gitproject/QTO
mkdir ${PROJ_ROOT}/measureTime

for compiler in ${backends[*]}
do
    echo "testing ${compiler}"
    OUTPUT_FILE=${PROJ_ROOT}/measureTime/output_${compiler}.txt
    CUDA_VISIBLE_DEVICES=0 ${PY_EXE} ${PROG} \
            --data_path data/FB15k-betae \
            --kbc_path kbc/FB15K/best_valid.model \
            --fraction 10 \
            --thrshd 0.001 \
            --neg_scale 6 \
            --compiler ${compiler} > ${OUTPUT_FILE}
done