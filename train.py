from models import *
from data import *
from utils import *
import gc
from pathlib import Path


num_workers = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nfolds = 20
TRAIN_FOLD = 0
SEED = 88
torch_fix_seed(seed=SEED)

# Set the desired path for saving models
model_save_path = Path("models/")
model_save_path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist


CFGX = {
    'block_size': 15000,
    'block_stride': 15000 // 4,
    'patch_size': 15,

    'sleepformer_dim': 320,
    'sleepformer_num_heads': 4,
    'sleepformer_num_encoder_layers': 4,
    'sleepformer_num_lstm_layers': 2,
    'sleepformer_lstm_dropout': 0.0,
    'sleepformer_first_dropout': 0.2,
    'sleepformer_encoder_dropout': 0.2,
    'sleepformer_mha_dropout': 0.2,
    'sleepformer_ffn_multiplier': 1,

    'fname': 'sleepformer-17',
    'batch_size': 32,
    'lr': 3e-4,
    'wd': 0.1,
    'epochs': 16,

}

CFG1 = {
    'block_size': 16000,
    'block_stride': 16000 // 4,
    'patch_size': 16,

    'sleepformer_dim': 240,
    'sleepformer_num_heads': 4,
    'sleepformer_num_encoder_layers': 3,
    'sleepformer_num_lstm_layers': 2,
    'sleepformer_lstm_dropout': 0.0,
    'sleepformer_first_dropout': 0.15,
    'sleepformer_encoder_dropout': 0.15,
    'sleepformer_mha_dropout': 0.15,
    'sleepformer_ffn_multiplier': 1,

    'fname': 'sleepformer-18',
    'batch_size': 18,
    'lr': 3e-4,
    'wd': 0.05,
    'epochs': 15,

}


CFG2 = {
    'block_size': 16000,
    'block_stride': 16000 // 4,
    'patch_size': 16,

    'sleepformer_dim': 240,
    'sleepformer_num_heads': 4,
    'sleepformer_num_encoder_layers': 3,
    'sleepformer_num_lstm_layers': 2,
    'sleepformer_lstm_dropout': 0.0,
    'sleepformer_first_dropout': 0.15,
    'sleepformer_encoder_dropout': 0.15,
    'sleepformer_mha_dropout': 0.15,
    'sleepformer_ffn_multiplier': 1,

    'fname': 'sleepformer-19',
    'batch_size': 18,
    'lr': 5e-4,
    'wd': 0.1,
    'epochs': 15,

}


CFG3 = {
    'block_size': 18000,
    'block_stride': 18000 // 4,
    'patch_size': 18,

    'sleepformer_dim': 192,
    'sleepformer_num_heads': 6,
    'sleepformer_num_encoder_layers': 5,
    'sleepformer_num_lstm_layers': 2,
    'sleepformer_lstm_dropout': 0.0,
    'sleepformer_first_dropout': 0.1,
    'sleepformer_encoder_dropout': 0.1,
    'sleepformer_mha_dropout': 0.0,
    'sleepformer_ffn_multiplier': 1,

    'fname': 'sleepformer-7',
    'batch_size': 18,
    'lr': 3e-4,
    'epochs': 15,

}


CFG4 = {
    'block_size': 12000,
    'block_stride': 12000 // 4, #This honestly determines the amount of data to be trained on 
    'patch_size': 12,

    'sleepformer_dim': 360,
    'sleepformer_num_heads': 4,
    'sleepformer_num_encoder_layers': 4,
    'sleepformer_num_lstm_layers': 2,
    'sleepformer_lstm_dropout': 0.0,
    'sleepformer_first_dropout': 0.15,
    'sleepformer_encoder_dropout': 0.15,
    'sleepformer_mha_dropout': 0.15,
    'sleepformer_ffn_multiplier': 1,

    'fname': 'sleepformer-19',
    'batch_size': 32,
    'lr': 3e-4,
    'wd': 0.1,
    'epochs': 15,
}


CFG5 = {
    'block_size': 17280,
    'block_stride': 17280 // 2, #This honestly determines the amount of data to be trained on 
    'patch_size': 12,

    'sleepformer_dim': 280,
    'sleepformer_num_heads': 4,
    'sleepformer_num_encoder_layers': 4,
    'sleepformer_num_lstm_layers': 2,
    'sleepformer_lstm_dropout': 0.0,
    'sleepformer_first_dropout': 0.1,
    'sleepformer_encoder_dropout': 0.1,
    'sleepformer_mha_dropout': 0.1,
    'sleepformer_ffn_multiplier': 1,
    'recycle_iterations': 3, 


    'fname': 'sleepformer-26',
    'batch_size': 32,
    'lr': 3e-4,
    'wd': 0.1,
    'epochs': 15,
    'zbp': 0.05, 

}



CFG6 = {
    'block_size': 17280,
    'block_stride': 17280 // 2, #This honestly determines the amount of data to be trained on 
    'patch_size': 24,

    'sleepformer_dim': 320,
    'sleepformer_num_heads': 4,
    'sleepformer_num_encoder_layers': 4,
    'sleepformer_num_lstm_layers': 2,
    'sleepformer_lstm_dropout': 0.0,
    'sleepformer_first_dropout': 0.1,
    'sleepformer_encoder_dropout': 0.1,
    'sleepformer_mha_dropout': 0.1,
    'sleepformer_ffn_multiplier': 1,
    'recycle_iterations': 3, 


    'fname': 'sleepformer-26',
    'batch_size': 32,
    'lr': 3e-4,
    'wd': 0.1,
    'epochs': 15,
    'zbp': 0.05, 

}


CFG7 =    {
    'block_size': 18000,
    'block_stride': 18000 // 12,
    'patch_size': 18,

    'sleepformer_dim': 320,
    'sleepformer_num_heads': 4,
    'sleepformer_num_encoder_layers': 4,
    'sleepformer_num_lstm_layers': 1,
    'sleepformer_lstm_dropout': 0.0,
    'sleepformer_first_dropout': 0.1,
    'sleepformer_encoder_dropout': 0.1,
    'sleepformer_mha_dropout': 0.0,
    'sleepformer_ffn_multiplier': 1,
    'recycle_iterations': 3, 

    'fname': 'sleepformer-30',
    'batch_size': 12,
    'lr': 5e-4,
    'wd': 0.1,
    'epochs': 30,
    'zbp': 0.02, 
}


CFG8 =    {
    'block_size': 16200,
    'block_stride': 16200 // 3,
    'patch_size': 18,

    'sleepformer_dim': 320,
    'sleepformer_num_heads': 4,
    'sleepformer_num_encoder_layers': 4,
    'sleepformer_num_lstm_layers': 1,
    'sleepformer_lstm_dropout': 0.0,
    'sleepformer_first_dropout': 0.1,
    'sleepformer_encoder_dropout': 0.1,
    'sleepformer_mha_dropout': 0.0,
    'sleepformer_ffn_multiplier': 1,
    'recycle_iterations': 3, 

    'fname': 'sleepformer-30',
    'batch_size': 28,
    'lr': 5e-4,
    'wd': 0.1,
    'epochs': 40,
    'zbp': 0.02, 
}



#cfgs = [CFG6, CFG7, CFGX, CFG1, CFG2, CFG4, CFG5]
cfgs = [CFG8, CFG7]


for CFG in cfgs: # running multiple folds at kaggle may cause OOM

    skf = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)
    metadata = pd.read_csv('train_events.csv')
    unique_ids = metadata['series_id'].unique()
    meta_cts = pd.DataFrame(unique_ids, columns=['series_id'])
    for i, (train_index, valid_index) in enumerate(skf.split(X=meta_cts['series_id'], y=[1]*len(meta_cts))):
        if i != TRAIN_FOLD:
            continue
        print(f"Fold = {i}")
        train_ids = meta_cts.loc[train_index, 'series_id']
        valid_ids = meta_cts.loc[valid_index, 'series_id']
        print(f"Length of Train = {len(train_ids)}, Length of Valid = {len(valid_ids)}")

        if i == TRAIN_FOLD:
            break

    train_fpaths = [f"/sleepformer/dataset/train_csvs/{_id}.csv" for _id in train_ids]
    valid_fpaths = [f"/sleepformer/dataset/train_csvs/{_id}.csv" for _id in valid_ids]
    print(f"Validation Samples: {valid_fpaths}")

    gc.collect()
    torch.cuda.empty_cache()

    print("Preparing Data...")
    ds_train = SleepDataset(train_fpaths, CFG, is_train=True, zbp=CFG['zbp'], target_thresh = None) #zbp = zero block percentage
    dl_train = torch.utils.data.DataLoader(ds_train, num_workers=num_workers, batch_size=CFG['batch_size'],
                persistent_workers=True, shuffle=True)
    dl_train = DeviceDataLoader(dl_train, device)


    ds_val = SleepDataset(valid_fpaths, CFG, is_train=False, zbp=CFG['zbp'], target_thresh = None)
    dl_val= torch.utils.data.DataLoader(ds_val, num_workers=num_workers, batch_size=CFG['batch_size'],
                                        persistent_workers=False, shuffle=False)
    dl_val = DeviceDataLoader(dl_val, device)

    gc.collect()
    data = DataLoaders(dl_train,dl_val)

    print("Preparing the model...")

    #model = SleepFormer(CFG).to(device)
    model = RSleepFormer(CFG).to(device)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

    loss = CustomLoss(CFG, z_loss_weight=1e-4)

    # Create the learner and add SaveModelCallback
    learn = Learner(data, model, path=model_save_path, loss_func=loss, opt_func=Adam, cbs=[GradientClip(3.0), SaveModelCallback(monitor='valid_loss', fname=CFG['fname'], with_opt=True)]).to_fp16()

    OUT = "models/"
    lr = CFG['lr']
    epochs = CFG['epochs']
    weight_decay = CFG['wd']
    

    # Train the model
    print(f"Training {CFG['fname']}")
    learn.fit_one_cycle(epochs, lr_max=lr, wd=weight_decay, pct_start=0.02)

    # Check if the best model was saved and push to GitHub
    best_model_path = model_save_path / 'models' / f'{CFG["fname"]}.pth'
    if os.path.exists(best_model_path):
        print("Training finished. Best model saved.")
        print(f"Pushing {best_model_path} to GitHub...")
        push_to_github(best_model_path, f"Added {CFG['fname']}!")
    else:
        print("No best model saved.")

    gc.collect()
    torch.cuda.empty_cache()


