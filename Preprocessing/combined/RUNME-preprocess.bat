@echo off
cd /d %~dp0
cmd /k "sklearn-env\Scripts\activate && py 1-remove.py && py 2-one_hot_enc.py && py 3-min_max.py && py 4-stratified_split.py && py 5-random_resample.py && py 6-feature_select.py"
