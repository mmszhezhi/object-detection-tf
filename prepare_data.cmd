@echo off
set images=annotations2/data/aug-a0
set dst=annotations2
set name=t1

python xml_to_csv.py %images% %dst%/%name%.csv
python csv2record.py --output_path_train=%dst%/%name%-train.record --output_path_test=%dst%/%name%-test.record --img_path=%images% --csv_input=%dst%/%name%.csv
