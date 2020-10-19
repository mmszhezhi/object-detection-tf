@echo off
set dir=t3
python model_main_tf2.py --model_dir=models/%dir% --pipeline_config_path=models/%dir%/pipeline.config --checkpoint_every_n=100 --num_workers=7



