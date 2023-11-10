from models import *
from data import *
from utils import *
import gc
from pathlib import Path


num_workers = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nfolds = 4
TRAIN_FOLD = 0
SEED = 88
torch_fix_seed(seed=SEED)

# Set the desired path for saving models
model_save_path = Path("models/")
model_save_path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist


CFG1 = {
    'block_size': 14400,
    'block_stride': 14400 // 8,
    'patch_size': 12,

    'sleepformer_dim': 24,
    'sleepformer_num_heads': 4,
    'sleepformer_num_encoder_layers': 6,
    'sleepformer_num_lstm_layers': 2,
    'sleepformer_lstm_dropout': 0.0,
    'sleepformer_first_dropout': 0.1,
    'sleepformer_encoder_dropout': 0.1,
    'sleepformer_mha_dropout': 0.1,
    'sleepformer_ffn_multiplier': 4,

    'fname': 'sleepformer-64_0',
    'batch_size': 42,
    'lr': 5e-4,
    'wd': 0.1,
    'epochs': 15,
    'zbp': 0.02,
    'fold': 0,
    
}


CFG2 = {
    'block_size': 14400,
    'block_stride': 14400 // 8,
    'patch_size': 12,

    'sleepformer_dim': 24,
    'sleepformer_num_heads': 4,
    'sleepformer_num_encoder_layers': 6,
    'sleepformer_num_lstm_layers': 2,
    'sleepformer_lstm_dropout': 0.0,
    'sleepformer_first_dropout': 0.1,
    'sleepformer_encoder_dropout': 0.1,
    'sleepformer_mha_dropout': 0.1,
    'sleepformer_ffn_multiplier': 4,

    'fname': 'sleepformer-64_1',
    'batch_size': 42,
    'lr': 5e-4,
    'wd': 0.1,
    'epochs': 15,
    'zbp': 0.02, 
    'fold': 1,

}


CFG3 = {
    'block_size': 14400,
    'block_stride': 14400 // 8,
    'patch_size': 12,

    'sleepformer_dim': 24,
    'sleepformer_num_heads': 4,
    'sleepformer_num_encoder_layers': 6,
    'sleepformer_num_lstm_layers': 2,
    'sleepformer_lstm_dropout': 0.0,
    'sleepformer_first_dropout': 0.1,
    'sleepformer_encoder_dropout': 0.1,
    'sleepformer_mha_dropout': 0.1,
    'sleepformer_ffn_multiplier': 4,

    'fname': 'sleepformer-64_2',
    'batch_size': 42,
    'lr': 5e-4,
    'wd': 0.1,
    'epochs': 15,
    'zbp': 0.02, 
    'fold': 2,

}


CFG4 = {
    'block_size': 14400,
    'block_stride': 14400 // 8,
    'patch_size': 12,

    'sleepformer_dim': 24,
    'sleepformer_num_heads': 4,
    'sleepformer_num_encoder_layers': 6,
    'sleepformer_num_lstm_layers': 2,
    'sleepformer_lstm_dropout': 0.0,
    'sleepformer_first_dropout': 0.1,
    'sleepformer_encoder_dropout': 0.1,
    'sleepformer_mha_dropout': 0.1,
    'sleepformer_ffn_multiplier': 4,

    'fname': 'sleepformer-64_3',
    'batch_size': 42,
    'lr': 5e-4,
    'wd': 0.1,
    'epochs': 15,
    'zbp': 0.02, 
    'fold': 3,

}


CFG5 = {
    'block_size': 14400,
    'block_stride': 14400 // 8,
    'patch_size': 12,
    #'block_size': 21600,
    #'block_stride': 21600 // 8,
    #'patch_size': 24,

    'sleepformer_dim': 24,
    'sleepformer_num_heads': 4,
    'sleepformer_num_encoder_layers': 6,
    'sleepformer_num_lstm_layers': 2,
    'sleepformer_lstm_dropout': 0.0,
    'sleepformer_first_dropout': 0.1,
    'sleepformer_encoder_dropout': 0.1,
    'sleepformer_mha_dropout': 0.1,
    'sleepformer_ffn_multiplier': 4,


    'fname': 'sleepformer-63',
    'batch_size': 24,
    'lr': 2e-4,
    'wd': 0.05,
    'epochs': 30,
    'zbp': 0.02, 

}


#CFG1, CFG2, CFG3, CFG7, 
#cfgs = [CFG7, CFG6, CFG5]
cfgs = [CFG2, CFG3, CFG4]


for CFG in cfgs: # running multiple folds at kaggle may cause OOM
    TRAIN_FOLD = CFG['fold']

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

    train_fpaths = [f"/sleepformer/dataset/kaggle/working/train_series/{_id}.parquet" for _id in train_ids]
    valid_fpaths = [f"/sleepformer/dataset/kaggle/working/train_series/{_id}.parquet" for _id in valid_ids]
    print(f"Validation Samples: {valid_fpaths}")

    #train_fpaths = train_fpaths[:4]
    #valid_fpaths = valid_fpaths[:1]
    
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
    model = SleepFormer(CFG).to(device)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

    loss = CustomLoss(CFG, z_loss_weight=1e-4)

    # Create the learner and add SaveModelCallback
    learn = Learner(data, model, path=model_save_path, loss_func=loss, opt_func=Adam, cbs=[GradientClip(3.0), SaveModelCallback(monitor='map', fname=CFG['fname'], with_opt=True)], metrics=[MAPMetric()]).to_fp16()

    OUT = "models/"
    lr = CFG['lr']
    epochs = CFG['epochs']
    weight_decay = CFG['wd']
    

    # Train the model
    print(f"Training {CFG['fname']}")
    learn.fit_one_cycle(epochs, lr_max=lr, wd=weight_decay, pct_start=0.01)

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


