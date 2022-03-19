python main.py -data_type linear
python main.py -data_type xor

#################
## Experiments ##
#################
## Different hidden_dim ##
#python main.py -exp_name hidden_dim -data_type xor -hidden_dim 10 -exp_title
#python main.py -exp_name hidden_dim -data_type xor -hidden_dim 20
#python main.py -exp_name hidden_dim -data_type xor -hidden_dim 30
#python main.py -exp_name hidden_dim -data_type xor -hidden_dim 40
#python main.py -exp_name hidden_dim -data_type xor -hidden_dim 50
#python main.py -exp_name hidden_dim -data_type xor -hidden_dim 60
#python main.py -exp_name hidden_dim -data_type xor -hidden_dim 70
#python main.py -exp_name hidden_dim -data_type xor -hidden_dim 80
#python main.py -exp_name hidden_dim -data_type xor -hidden_dim 90
#python main.py -exp_name hidden_dim -data_type xor -hidden_dim 100

#python main.py -exp_name hidden_dim -data_type linear -hidden_dim 10 -exp_title
#python main.py -exp_name hidden_dim -data_type linear -hidden_dim 20
#python main.py -exp_name hidden_dim -data_type linear -hidden_dim 30
#python main.py -exp_name hidden_dim -data_type linear -hidden_dim 40
#python main.py -exp_name hidden_dim -data_type linear -hidden_dim 50
#python main.py -exp_name hidden_dim -data_type linear -hidden_dim 60
#python main.py -exp_name hidden_dim -data_type linear -hidden_dim 70
#python main.py -exp_name hidden_dim -data_type linear -hidden_dim 80
#python main.py -exp_name hidden_dim -data_type linear -hidden_dim 90
#python main.py -exp_name hidden_dim -data_type linear -hidden_dim 100

## Different lr ##
#python main.py -exp_name lr -data_type linear -lr 1e-1 -exp_title
#python main.py -exp_name lr -data_type linear -lr 5e-2
#python main.py -exp_name lr -data_type linear -lr 1e-2
#python main.py -exp_name lr -data_type linear -lr 5e-3

#python main.py -exp_name lr -data_type xor -lr 1e-1 -exp_title
#python main.py -exp_name lr -data_type xor -lr 5e-2
#python main.py -exp_name lr -data_type xor -lr 1e-2
#python main.py -exp_name lr -data_type xor -lr 5e-3

## Different activation ##
#python main.py -exp_name activation -data_type linear -activation none -exp_title
#python main.py -exp_name activation -data_type xor    -activation none
#python main.py -exp_name activation -data_type linear -activation relu -exp_title
#python main.py -exp_name activation -data_type xor    -activation relu