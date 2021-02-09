LR="0.02"
OPT="adagrad"
EPOCH="1"
GPU=0
log_number=0

# change libpath to the path where DeepCTR-Torch is installed
libpath=/home/local/mu/zd2922/miniconda3/lib/python3.8/site-packages/deepctr_torch/
# replace source code to enable MLET
cp ../utils/MLETbasemodel.py ${libpath}models/basemodel.py
cp ../utils/MLETcore.py ${libpath}layers/core.py
cp ../utils/MLETinputs.py ${libpath}inputs.py
cp ../models/MLETdcn.py ${libpath}models/dcn.py

logRootDir=MLETresults
logDir=$logRootDir/dcn

mkdir $logRootDir
mkdir $logDir

inner="64 0" # inner dimension for MLET
OUTER=32 # embedding dimension
for INNER in $inner
do
  echo "k${INNER}d${OUTER}"
  for lr in $LR
  do
    for opt in $OPT
    do
      for epoch in $EPOCH
      do
        echo "opt ${opt} lr ${lr} epoch ${epoch}"
        python3 train_dcn.py \
                --inner_dim $INNER --embd_dim $OUTER --batch 256 \
                --optimizer $opt --lr $lr --epoch $epoch --gpu_number $GPU \
                > $logDir/k${INNER}d${OUTER}_${opt}${lr}Ep${epoch}.log${log_number}
      done
    done
  done
done

inner="64 32 0"
OUTER=16
for INNER in $inner
do
  echo "k${INNER}d${OUTER}"
  for lr in $LR
  do
    for opt in $OPT
    do
      for epoch in $EPOCH
      do
        echo "opt ${opt} lr ${lr} epoch ${epoch}"
        python3 train_dcn.py \
                --inner_dim $INNER --embd_dim $OUTER --batch 256 \
                --optimizer $opt --lr $lr --epoch $epoch --gpu_number $GPU \
                > $logDir/k${INNER}d${OUTER}_${opt}${lr}Ep${epoch}.log${log_number}
      done
    done
  done
done

inner="64 32 16 0"
OUTER=8
for INNER in $inner
do
  echo "k${INNER}d${OUTER}"
  for lr in $LR
  do
    for opt in $OPT
    do
      for epoch in $EPOCH
      do
        echo "opt ${opt} lr ${lr} epoch ${epoch}"
        python3 train_dcn.py \
                --inner_dim $INNER --embd_dim $OUTER --batch 256 \
                --optimizer $opt --lr $lr --epoch $epoch --gpu_number $GPU \
                > $logDir/k${INNER}d${OUTER}_${opt}${lr}Ep${epoch}.log${log_number}
      done
    done
  done
done

inner="64 32 16 8 0"
OUTER=4
for INNER in $inner
do
  echo "k${INNER}d${OUTER}"
  for lr in $LR
  do
    for opt in $OPT
    do
      for epoch in $EPOCH
      do
        echo "opt ${opt} lr ${lr} epoch ${epoch}"
        python3 train_dcn.py \
                --inner_dim $INNER --embd_dim $OUTER --batch 256 \
                --optimizer $opt --lr $lr --epoch $epoch --gpu_number $GPU \
                > $logDir/k${INNER}d${OUTER}_${opt}${lr}Ep${epoch}.log${log_number}
      done
    done
  done
done
