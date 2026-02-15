# Compare with other models
set up 
- download weight 
```
bash download_weight.sh
```

then check path and run profling.sh
```
profiling.sh
or run python first cell on compare_zip.ipynb (can copy the code to colab session then fix the path)
```
output gonna at "/unifolm-world-model-action/compare/{device_name}" 

zip the entire output folder device_name by running the compare_zip.ipynb file 
```
zip_fast(f'./unifolm-world-model-action/compare/unitree_z1_stackbox/{device_name}', f'./{device_name}-compare.zip') 
```
then download and send the zip file to Run thanks.