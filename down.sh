export CUDA_VISIBLE_DEVICES=1
model_name=TSMixer
channel_function=RNN
temporal_function=patch


for len in    96 192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity \
    --data_path electricity.csv \
    --model_id ECL_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --patch 1 4 12 24 \
    --dropout 0.1 \
    --d2 0.1 \
    --channel_function  $channel_function \
    --temporal_function  $temporal_function \
    --down_sampling_layers 5 \
    --learning_rate 0.005 \
    --d_model 512 \
    --d_core 128 \
    --d_ff 1024 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --batch_size 32 \
    --itr 1
done


# Traffic
for len in  96 192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/traffic \
    --data_path traffic.csv \
    --model_id traffic_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --patch  1 4 12 24 \
    --dropout 0.1 \
    --d2 0.1 \
    --channel_function  $channel_function \
    --temporal_function  $temporal_function \
    --down_sampling_layers 5 \
    --d_model 512 \
    --d_core 128 \
    --d_ff 1024 \
    --learning_rate 0.005 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --lradj type75 \
    --train_epochs 20 \
    --patience 5 \
    --batch_size 32 \
    --itr 1
done

 # # weather 
for len in  96 192  336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather \
    --data_path weather.csv \
    --model_id weather_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --patch 1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.5 \
    --d2 0.5 \
    --channel_function  $channel_function \
    --temporal_function  $temporal_function \
    --down_sampling_layers 5 \
    --learning_rate 0.001 \
    --d_model 512 \
    --d_ff 512 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1
done


# ETTh1
for len in 96 192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_96 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --patch 1 4 12 24  \
    --pred_len $len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.5 \
    --d2 0.8 \
    --channel_function  $channel_function \
    --temporal_function  $temporal_function \
    --down_sampling_layers 5 \
    --learning_rate 0.001 \
    --d_model 512 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1
done



# # # ETTh2 
for len in 96 192 
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTh2.csv \
    --model_id ETTh2_96_96 \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len 96 \
    --patch 1 2 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.7 \
    --d2 0.7 \
    --channel_function  $channel_function \
    --temporal_function  $temporal_function \
    --down_sampling_layers 5 \
    --learning_rate 0.001 \
    --d_model 512 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1
done
for len in   336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTh2.csv \
    --model_id ETTh2_96_96 \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len 96 \
    --patch 1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.8 \
    --d2 0.8 \
    --channel_function  $channel_function \
    --temporal_function  $temporal_function \
    --down_sampling_layers 5 \
    --learning_rate 0.001 \
    --d_model 512 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1
done

# ETTm1

for len in  96 192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_96 \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --patch   1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.5 \
    --d2 0.8 \
    --channel_function  $channel_function \
    --temporal_function  $temporal_function \
    --down_sampling_layers 5 \
    --learning_rate 0.001 \
    --d_model 512 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1
done


# # ETTm2
for len in 96 192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTm2.csv \
    --model_id ETTm2_96_96 \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len 96  \
    --patch  1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.5 \
    --d2 0.8 \
    --channel_function  $channel_function \
    --temporal_function  $temporal_function \
    --down_sampling_layers 5 \
    --learning_rate 0.001 \
    --d_model 512 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1
done

for len in  96 192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Solar \
    --data_path solar_AL.txt \
    --model_id Solar_96_96 \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len 96 \
    --patch 1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.1 \
    --d2 0.1 \
    --channel_function  $channel_function \
    --temporal_function  $temporal_function \
    --down_sampling_layers 5 \
    --learning_rate 0.001 \
    --d_model 512 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --itr 1 \
    --use_norm 0
done




for len in  96  
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity \
    --data_path electricity.csv \
    --model_id ECL_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 576 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --patch 1  4 12  24 \
    --dropout 0.5 \
    --d2 0.5 \
    --down_sampling_layers 5 \
    --learning_rate 0.001 \
    --d_model 1024 \
    --d_core 128 \
    --d_ff 512 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --lradj type1 \
    --train_epochs 10 \
    --patience 3 \
    --batch_size 32 \
    --itr 1
done
for len in    192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity \
    --data_path electricity.csv \
    --model_id ECL_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --patch 1  4 12  24 \
    --dropout 0.5 \
    --d2 0.5 \
    --down_sampling_layers 5 \
    --learning_rate 0.001 \
    --d_model 1024 \
    --d_core 128 \
    --d_ff 512 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --lradj type1 \
    --train_epochs 10 \
    --patience 3 \
    --batch_size 32 \
    --itr 1
done






# # Traffic
for len in  96 192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/traffic \
    --data_path traffic.csv \
    --model_id traffic_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --label_len 48 \
    --pred_len $len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --patch 1  4 12 24 \
    --dropout 0.1 \
    --d2 0.1 \
    --down_sampling_layers 5 \
    --d_model 512 \
    --d_core 128 \
    --d_ff 1024 \
    --learning_rate 0.001 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --train_epochs 10 \
    --patience 3 \
    --batch_size 16 \
    --itr 1
done





#  weather 
for len in  96  720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather \
    --data_path weather.csv \
    --model_id weather_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 336 \
    --patch 1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.8 \
    --d2 0.8 \
    --down_sampling_layers 5 \
    --learning_rate 0.0001 \
    --d_model 512 \
    --d_ff 512 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1
done
for len in 192 336
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather \
    --data_path weather.csv \
    --model_id weather_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 336 \
    --patch 1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.5 \
    --d2 0.8 \
    --down_sampling_layers 5 \
    --learning_rate 0.0005 \
    --d_model 512 \
    --d_ff 512 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1
done






for len in  96 192
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_96 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 576 \
    --label_len 96 \
    --patch 1 4 12 24  \
    --pred_len $len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.8 \
    --d2 0.8 \
    --down_sampling_layers 5 \
    --learning_rate 0.0005 \
    --d_model 512 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1
done


for len in  336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_96 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 336 \
    --label_len 96 \
    --patch 1 4 12 24  \
    --pred_len $len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.8 \
    --d2 0.8 \
    --down_sampling_layers 5 \
    --learning_rate 0.0005 \
    --d_model 512 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1
done



for len in 96  
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTh2.csv \
    --model_id ETTh2_96_96 \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len 336 \
    --patch  1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.8 \
    --d2 0.8 \
    --down_sampling_layers 5 \
    --learning_rate 0.0005 \
    --d_model 512 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1
done

for len in 96 192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTh2.csv \
    --model_id ETTh2_96_96 \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len 576 \
    --patch  1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.9 \
    --d2 0.9 \
    --down_sampling_layers 1 \
    --learning_rate 0.0005 \
    --d_model 512 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1
done




for len in 96 192  
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_96 \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 576 \
    --patch   1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.8 \
    --d2 0.8 \
    --down_sampling_layers 5 \
    --learning_rate 0.001 \
    --d_model 512 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1
done

for len in  336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_96 \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 576 \
    --patch   1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.8 \
    --d2 0.8 \
    --down_sampling_layers 5 \
    --learning_rate 0.001 \
    --d_model 512 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1
done
for len in    720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_96 \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 720 \
    --patch   1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.9 \
    --d2 0.9 \
    --down_sampling_layers 3 \
    --learning_rate 0.0001 \
    --d_model 512 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1
done





for len in  96 192 336   
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTm2.csv \
    --model_id ETTm2_96_96 \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len 576  \
    --patch   1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.8 \
    --d2 0.8 \
    --down_sampling_layers 5 \
    --learning_rate 0.0001 \
    --d_model 512 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1
done

for len in   336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTm2.csv \
    --model_id ETTm2_96_96 \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len 720  \
    --patch   1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.8 \
    --d2 0.8 \
    --down_sampling_layers 5 \
    --learning_rate 0.0001 \
    --d_model 512 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1
done
for len in     720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTm2.csv \
    --model_id ETTm2_96_96 \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len 720  \
    --patch   1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.9 \
    --d2 0.9 \
    --down_sampling_layers 3 \
    --learning_rate 0.0001 \
    --d_model 512 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1
done





for len in  96  
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Solar \
    --data_path solar_AL.txt \
    --model_id Solar_96_96 \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len 192 \
    --patch   1  12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.5 \
    --d2 0.5 \
    --down_sampling_layers 5 \
    --learning_rate 0.001 \
    --d_model 512 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --itr 1 \
    --use_norm 0
done

for len in    192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Solar \
    --data_path solar_AL.txt \
    --model_id Solar_96_96 \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len 576 \
    --patch  1  12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.8 \
    --d2 0.8 \
    --down_sampling_layers 5 \
    --learning_rate 0.001 \
    --d_model 512 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --itr 1 \
    --use_norm 0
done
