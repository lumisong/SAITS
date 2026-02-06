# python run experiments for NewZealand dataset with BRITS
python run_models.py \
    --config_path local_workspace/configs/NewZealand/NewZealand_BRITS_best.ini

# python run experiments for NewZealand dataset with MRNN
python run_models.py \
    --config_path local_workspace/configs/NewZealand/NewZealand_MRNN_best.ini

# python run experiments for NewZealand dataset with Transformer
python run_models.py \
    --config_path local_workspace/configs/NewZealand/NewZealand_Transformer_best.ini

# python run experiments for NewZealand dataset with SAITS base
python run_models.py \
    --config_path local_workspace/configs/NewZealand/NewZealand_SAITS_base.ini

# python run experiments for NewZealand dataset with SAITS best
python run_models.py \
    --config_path local_workspace/configs/NewZealand/NewZealand_SAITS_best.ini
