@echo off
set testset=testimg
set model_path=models/t2/graph/saved_model
set PATH_TO_LABELS=annotations/labelmap.pbtxt
set save_result=testresult1
set NUM_CLASSES=10
python evaluatev2.py --testset=%testset% --model_path=%model_path% --PATH_TO_LABELS=%PATH_TO_LABELS% --NUM_CLASSES=%NUM_CLASSES% --save_result=%save_result%