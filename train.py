from models import *
from data import *
from utils import *
import gc

num_workers = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nfolds = 6
TRAIN_FOLD = 0
SEED = 27
torch_fix_seed(seed=SEED)

CFG = {
    #'block_size': 17280,
    #'block_stride': 17280 // 16,
    #'patch_size': 12,
    'block_size': 18000,
    'block_stride': 18000 // 8,
    'patch_size': 24,

    'sleepformer_dim': 192,
    'sleepformer_num_heads': 6,
    'sleepformer_num_encoder_layers': 5,
    'sleepformer_num_lstm_layers': 2,
    'sleepformer_lstm_dropout': 0.0,
    'sleepformer_first_dropout': 0.1,
    'sleepformer_encoder_dropout': 0.1,
    'sleepformer_mha_dropout': 0.0,
    'sleepformer_ffn_multiplier': 1,

    'fname': 'sleepformer-3',
    'batch_size': 24,
    'lr': 2.5e-3,
    'epochs': 10,

}


for fold in [0]: # running multiple folds at kaggle may cause OOM

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


    print("Preparing Data...")
    ds_train = SleepDataset(train_fpaths, CFG, is_train=True, zbp=0.20) #zbp = zero block percentage
    dl_train = torch.utils.data.DataLoader(ds_train, num_workers=num_workers, batch_size=CFG['batch_size'],
                persistent_workers=True, shuffle=True)
    dl_train = DeviceDataLoader(dl_train, device)


    ds_val = SleepDataset(valid_fpaths, CFG, is_train=False, zbp=0.20)
    dl_val= torch.utils.data.DataLoader(ds_val, num_workers=num_workers, batch_size=CFG['batch_size'],
                                        persistent_workers=False, shuffle=False)
    dl_val = DeviceDataLoader(dl_val, device)

    gc.collect()
    data = DataLoaders(dl_train,dl_val)

    print("Preparing the model...")

    model = SleepFormer(CFG).to(device)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

    loss = CustomLoss(CFG, z_loss_weight=1e-4)

    learn = Learner(data, model, loss_func=loss,cbs=[GradientClip(3.0)]).to_fp16() #,
                #metrics=[ED_AP]).to_fp16() #metrics=[ED_AP]

    lr = CFG['lr']
    epochs = CFG['epochs']

    learn.fit_one_cycle(epochs, lr_max=lr, wd=0.05, pct_start=0.03)
    torch.save(learn.model.state_dict(),os.path.join(OUT,f'{fname}.pth'))
    gc.collect()


    OUT = "models/"
    #fname = "rnaformer-90"
    fname = CFG["fname"]

    print("Training finished. Saving the model...")
    model_path = os.path.join(OUT, f'{fname}.pth')
    torch.save(learn.model.state_dict(), model_path)
        
    torch.cuda.empty_cache()
    gc.collect()

    # Push model and stats to GitHub
    print(f"Pushing {fname} model to GitHub...")
    push_to_github(model_path, f"Added {fname}!")


