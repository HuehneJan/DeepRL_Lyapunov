# Create logging dir
mkdir -p ./loggs
printf "Starting Nonlinear System Analysis Experiments \n\n" > ./loggs/progress_information.txt

# Linear NN Controller
now=$(date)
printf "Linear NN Controller Training. Started at: $s \n "%now >> ./loggs/progress_information.txt
python3  train.py --save_dir=linear --run_name=linear --mass=1 --length=1 --use_linear > ./loggs/linear_nn.txt
python3 average.py --save_dir=linear --model_dir=linear --mass=1 --length=1 --use_linear >> ./loggs/linear_nn.txt
now=$(date)
printf "Linear NN Controller Training. Ended at: $s \n\n "%now >> ./loggs/progress_information.txt


# Nonlinear DNN Controller
now=$(date)
printf "Nonlinear DNN Controller Training. Started at: $s \n "%now >> ./loggs/progress_information.txt
python3  train.py --save_dir=nonlinear --run_name=nonlinear --mass=1 --length=1 > ./loggs/nonlinear_dnn.txt
python3 average.py --save_dir=nonlinear --model_dir=nonlinear --mass=1 --length=1 >> ./loggs/nonlinear_dnn.txt
python3 linearize.py --model_dir=nonlinear --mass=1 --length=1 >> ./loggs/nonlinear_dnn.txt
python3 roa_estimation.py --model_dir=nonlinear --mass=1 --length=1 >> ./loggs/nonlinear_dnn.txt
now=$(date)
printf "Nonlinear DNN Controller Training. Ended at: $s \n\n "%now >> ./loggs/progress_information.txt

# Comparison
now=$(date)
printf "Controller Comparison. Started at: $s \n "%now >> ./loggs/progress_information.txt
python3 compare.py > ./loggs/comparison.txt
#Parameter Variation 1
now=$(date)
printf "Parameter Variation 1 Training. Started at: $s \n "%now >> ./loggs/progress_information.txt
printf "Model Variation Run 1. Mass=2kg    Length=1m   Max Torque=2Nm  \n\n" > ./loggs/model_var_1.txt
python3  train.py --save_dir=model_var/1 --run_name=model_var_1 --mass=2 --length=1 >> ./loggs/model_var_1.txt
python3 average.py --save_dir=model_var/1 --model_dir=model_var/1 --mass=2 --length=1 >> ./loggs/model_var_1.txt
python3 linearize.py --model_dir=model_var/1 --mass=2 --length=1 >> ./loggs/model_var_1.txt
python3 roa_estimation.py --model_dir=model_var/1 --mass=2 --length=1 >> ./loggs/model_Var_1.txt
now=$(date)
printf "Parameter Variation 1 Training. Ended at: $s \n "%now >> ./loggs/progress_information.txt


#Parameter Variation 2
now=$(date)
printf "Parameter Variation 2 Training. Started at: $s \n "%now >> ./loggs/progress_information.txt
printf "Model Variation Run 2. Mass=4kg    Length=1m   Max Torque=2Nm  \n\n" > ./loggs/model_var_2.txt
python3  train.py --save_dir=model_var/2 --run_name=model_var_2 --mass=4 --length=1 >> ./loggs/model_var_2.txt
python3 average.py --save_dir=model_var/2 --model_dir=model_var/2 --mass=4 --length=1 >> ./loggs/model_var_2.txt
python3 linearize.py --model_dir=model_var/2 --mass=4 --length=1 >> ./loggs/model_var_2.txt
python3 roa_estimation.py --model_dir=model_var/2 --mass=4 --length=1 >> ./loggs/model_var_2.txt
now=$(date)
printf "Parameter Variation 2 Training. Ended at: $s \n "%now >> ./loggs/progress_information.txt


#Parameter Variation 3
now=$(date)
printf "Parameter Variation 3 Training. Started at: $s \n "%now >> ./loggs/progress_information.txt
printf "Model Variation Run 3. Mass=6kg    Length=1m   Max Torque=2Nm  \n\n" > ./loggs/model_var_3.txt
python3  train.py --save_dir=model_var/3 --run_name=model_var_3 --mass=6 --length=1 >> ./loggs/model_var_3.txt
python3 average.py --save_dir=model_var/3 --model_dir=model_var/3 --mass=6 --length=1 >> ./loggs/model_var_3.txt
python3 linearize.py --model_dir=model_var/3 --mass=6 --length=1 >> ./loggs/model_var_3.txt
python3 roa_estimation.py --model_dir=model_var/3 --mass=6 --length=1 >> ./loggs/model_var_3.txt
now=$(date)
printf "Parameter Variation 3 Training. Ended at: $s \n "%now >> ./loggs/progress_information.txt


#Parameter Variation 4
now=$(date)
printf "Parameter Variation 4 Training. Started at: $s \n "%now >> ./loggs/progress_information.txt
printf "Model Variation Run 4. Mass=1kg    Length=2m   Max Torque=2Nm  \n\n" > ./loggs/model_var_4.txt
python3  train.py --save_dir=model_var/4 --run_name=model_var_4 --mass=1 --length=2 >> ./loggs/model_var_4.txt
python3 average.py --save_dir=model_var/4 --model_dir=model_var/4 --mass=1 --length=2 >> ./loggs/model_var_4.txt
python3 linearize.py --model_dir=model_var/4 --mass=1 --length=2 >> ./loggs/model_var_4.txt
python3 roa_estimation.py --model_dir=model_var/4 --mass=1 --length=2 ./loggs/model_var_4.txt
now=$(date)
printf "Parameter Variation 4 Training. Ended at: $s \n "%now >> ./loggs/progress_information.txt


#Parameter Variation 5
now=$(date)
printf "Parameter Variation 5 Training. Started at: $s \n "%now >> ./loggs/progress_information.txt
printf "Model Variation Run 5. Mass=1kg    Length=4m   Max Torque=2Nm  \n\n" > ./loggs/model_var_5.txt
python3  train.py --save_dir=model_var/5 --run_name=model_var_5 --mass=1 --length=4 >> ./loggs/model_var_5.txt
python3 average.py --save_dir=model_var/5 --model_dir=model_var/5 --mass=1 --length=4 >> ./loggs/model_var_5.text
python3 linearize.py --model_dir=model_var/5 --mass=1 --length=4 >> ./loggs/model_var_5.txt
python3 roa_estimation.py --model_dir=model_var/5 --mass=1 --length=4 >> ./loggs/model_var_5.txt
now=$(date)
printf "Parameter Variation 5 Training. Ended at: $s \n "%now >> ./loggs/progress_information.txt


#Parameter Variation 6
now=$(date)
printf "Parameter Variation 6 Training. Started at: $s \n "%now >> ./loggs/progress_information.txt
printf "Model Variation Run 6. Mass=2kg    Length=2m   Max Torque=2Nm  \n\n" > ./loggs/model_var_6.txt
python3  train.py --save_dir=model_var/6 --run_name=model_var_6 --mass=2 --length=2 >> ./loggs/model_var_6.txt
python3 average.py --save_dir=model_var/6 --model_dir=model_var/6 --mass=2 --length=2 >> ./loggs/model_var_6.txt
python3 linearize.py --model_dir=model_var/6 --mass=2 --length=2 >> ./loggs/model_var_6.txt
python3 roa_estimation.py --model_dir=model_var/6 --mass=2 --length=2 >> ./loggs/model_var_6.txt
now=$(date)
printf "Parameter Variation 6 Training. Ended at: $s \n "%now >> ./loggs/progress_information.txt


#Parameter Variation 7
now=$(date)
printf "Parameter Variation 7 Training. Started at: $s \n "%now >> ./loggs/progress_information.txt
printf "Model Variation Run 7. Mass=0.25kg    Length=0.75m   Max Torque=2Nm  \n\n" > ./loggs/model_var_7.txt
python3  train.py --save_dir=model_var/7 --run_name=model_var_7 --mass=0.25 --length=0.75 >> ./loggs/model_var_7.txt
python3 average.py --save_dir=model_var/7 --model_dir=model_var/7 --mass=0.25 --length=0.75 >> ./loggs/model_var_7.txt
python3 linearize.py --model_dir=model_var/7 --mass=0.25 --length=0.75 >> ./loggs/model_var_7.txt
python3 roa_estimation.py --model_dir=model_var/7 --mass=0.25 --length=0.75 >> ./loggs/model_var_7.txt
now=$(date)
printf "Parameter Variation 7 Training. Ended at: $s \n "%now >> ./loggs/progress_information.txt


#Parameter Variation 8
now=$(date)
printf "Parameter Variation 8 Training. Started at: $s \n "%now >> ./loggs/progress_information.txt
printf "Model Variation Run 8. Mass=0.75kg    Length=0.5m   Max Torque=2Nm  \n\n" > ./loggs/model_var_8.txt
python3  train.py --save_dir=model_var/8 --run_name=model_var_8 --mass=0.75 --length=0.5  >> ./loggs/model_var_8.txt
python3 average.py --save_dir=model_var/8 --model_dir=model_var/8 --mass=0.75 --length=0.5  >> ./loggs/model_var_8.txt
python3 linearize.py --model_dir=model_var/8 --mass=0.75 --length=0.5  >> ./loggs/model_var_8.txt
python3 roa_estimation.py --model_dir=model_var/8 --mass=0.75 --length=0.5  >> ./loggs/model_var_8.txt
now=$(date)
printf "Parameter Variation 8 Training. Ended at: $s \n "%now >> ./loggs/progress_information.txt


#Parameter Variation 9
now=$(date)
printf "Parameter Variation 9 Training. Started at: $s \n "%now >> ./loggs/progress_information.txt
printf "Model Variation Run 9. Mass=6kg    Length=1m   Max Torque=4Nm  \n\n" > ./loggs/model_var_9.txt
python3  train.py --save_dir=model_var/9 --run_name=model_var_9 --mass=6 --length=1 --max_action=4.0 >> ./loggs/model_var_9.txt
python3 average.py --save_dir=model_var/9 --model_dir=model_var/9 --mass=6 --length=1 --max_action=4.0 >> ./loggs/model_var_9.txt
python3 linearize.py --model_dir=model_var/9 --mass=6 --length=1 --max_action=4.0 >> ./loggs/model_var_9.txt
python3 roa_estimation.py --model_dir=model_var/9 --mass=6 --length=1 --max_action=4.0 >> ./loggs/model_var_9.txt
now=$(date)
printf "Parameter Variation 9 Training. Ended at: $s \n "%now >> ./loggs/progress_information.txt


#Parameter Variation 10
now=$(date)
printf "Parameter Variation 10 Training. Started at: $s \n "%now >> ./loggs/progress_information.txt
printf "Model Variation Run 10. Mass=1kg    Length=4m   Max Torque=4Nm  \n\n" > ./loggs/model_var_10.txt
python3  train.py --save_dir=model_var/10 --run_name=model_var_10 --mass=1 --length=4 --max_action=4.0 >> ./loggs/model_var_10.txt
python3 average.py --save_dir=model_var/10 --model_dir=model_var/10 --mass=1 --length=4 --max_action=4.0 >> ./loggs/model_var_10.txt
python3 linearize.py --model_dir=model_var/10 --mass=1 --length=4 --max_action=4.0 >> ./loggs/model_var_10.txt
python3 roa_estimation.py --model_dir=model_var/10 --mass=1 --length=4 --max_action=4.0 >> ./loggs/model_var_10.txt
now=$(date)
printf "Parameter Variation 10 Training. Ended at: $s \n "%now >> ./loggs/progress_information.txt


#Parameter Variation 11
now=$(date)
printf "Parameter Variation 11 Training. Started at: $s \n "%now >> ./loggs/progress_information.txt
printf "Model Variation Run 11. Mass=0.25kg    Length=0.75m   Max Torque=1Nm  \n\n" > ./loggs/model_var_11.txt
python3  train.py --save_dir=model_var/11 --run_name=model_var_11 --mass=0.25 --length=0.75 --max_action=1.0 >> ./loggs/model_var_11.txt
python3 average.py --save_dir=model_var/11 --model_dir=model_var/11 --mass=0.25 --length=0.75 --max_action=1.0 >> ./loggs/model_var_11.txt
python3 linearize.py --model_dir=model_var/11 --mass=0.25 --length=0.75 --max_action=1.0 >> ./loggs/model_var_11.txt
python3 roa_estimation.py --model_dir=model_var/11 --mass=0.25 --length=0.75 --max_action=1.0 >> ./loggs/model_var_11.txt
now=$(date)
printf "Parameter Variation 11 Training. Ended at: $s \n "%now >> ./loggs/progress_information.txt


#Parameter Variation 12
now=$(date)
printf "Parameter Variation 12 Training. Started at: $s \n "%now >> ./loggs/progress_information.txt
printf "Model Variation Run 12. Mass=0.75kg    Length=0.5m   Max Torque=1Nm  \n\n" > ./loggs/model_var_12.txt
python3  train.py --save_dir=model_var/12 --run_name=model_var_12 --mass=0.75 --length=0.5 --max_action=1.0 >> ./loggs/model_var_12.txt
python3 average.py --save_dir=model_var/12 --model_dir=model_var/12 --mass=0.75 --length=0.5 --max_action=1.0  >> ./loggs/model_var_12.txt
python3 linearize.py --model_dir=model_var/12 --mass=0.75 --length=0.5 --max_action=1.0  >> ./loggs/model_var_12.txt
python3 roa_estimation.py --model_dir=model_var/112 --mass=0.75 --length=0.5 --max_action=1.0  >> ./loggs/model_var_12.txt
now=$(date)
printf "Parameter Variation 12 Training. Ended at: $s \n "%now >> ./loggs/progress_information.txt

