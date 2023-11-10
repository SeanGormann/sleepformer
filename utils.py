import torch
from fastai.metrics import Metric
from sklearn.metrics import average_precision_score
from fastai.vision.all import *
from fastai.callback.tracker import SaveModelCallback
import subprocess



# Fix fastai bug to enable fp16 training with dictionaries

def flatten(o):
    "Concatenate all collections and items as a generator"
    for item in o:
        if isinstance(o, dict): yield o[item]; continue
        elif isinstance(item, str): yield item; continue
        try: yield from flatten(item)
        except TypeError: yield item

from torch.cuda.amp import GradScaler, autocast
@delegates(GradScaler)
class MixedPrecision(Callback):
    "Mixed precision training using Pytorch's `autocast` and `GradScaler`"
    order = 10
    def __init__(self, **kwargs): self.kwargs = kwargs
    def before_fit(self):
        self.autocast,self.learn.scaler,self.scales = autocast(),GradScaler(**self.kwargs),L()
    def before_batch(self): self.autocast.__enter__()
    def after_pred(self):
        if next(flatten(self.pred)).dtype==torch.float16: self.learn.pred = to_float(self.pred)
    def after_loss(self): self.autocast.__exit__(None, None, None)
    def before_backward(self): self.learn.loss_grad = self.scaler.scale(self.loss_grad)
    def before_step(self):
        "Use `self` as a fake optimizer. `self.skipped` will be set to True `after_step` if gradients overflow. "
        self.skipped=True
        self.scaler.step(self)
        if self.skipped: raise CancelStepException()
        self.scales.append(self.scaler.get_scale())
    def after_step(self): self.learn.scaler.update()

    @property
    def param_groups(self):
        "Pretend to be an optimizer for `GradScaler`"
        return self.opt.param_groups
    def step(self, *args, **kwargs):
        "Fake optimizer step to detect whether this batch was skipped from `GradScaler`"
        self.skipped=False
    def after_fit(self): self.autocast,self.learn.scaler,self.scales = None,None,None

import fastai
fastai.callback.fp16.MixedPrecision = MixedPrecision




def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms = True
    # torch.backends.cudnn.benchmark = True





### Faster Metric, May crash my everything

class ED_AP(Metric):
    def __init__(self, tolerances):
        self.tolerances = tolerances
        self.reset()

    def reset(self):
        self.batch_ap_scores = []

    def accumulate(self, learn):
        preds = learn.pred
        targets = learn.yb[0]['Y'].cpu()

        preds = torch.sigmoid(preds).cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        batch_ap_scores = []
        for tolerance in self.tolerances:
            matched_preds, matched_targets = self.match_predictions(preds, targets, tolerance)
            ap = average_precision_score(matched_targets, matched_preds)
            batch_ap_scores.append(ap)

        # Store the average AP score for the batch
        self.batch_ap_scores.append(sum(batch_ap_scores) / len(batch_ap_scores))

        # Optional: Clear cache
        #torch.cuda.empty_cache()

    @property
    def value(self):
        return sum(self.batch_ap_scores) / len(self.batch_ap_scores)

    def match_predictions(self, predictions, targets, tolerance):
        """
        Match predictions with targets within the given tolerance.

        Args:
            predictions (numpy.ndarray): Array of predicted events. Flattened.
            targets (numpy.ndarray): Array of actual events. Flattened.
            tolerance (int): Tolerance for matching predictions with targets.

        Returns:
            matched_preds (numpy.ndarray): Matched predictions.
            matched_targets (numpy.ndarray): Matched targets.
        """
        # Check if the predictions are within tolerance of the targets
        within_tolerance = np.abs(predictions - targets) <= tolerance

        # Select predictions and targets that are within the tolerance
        matched_preds = predictions[within_tolerance]
        matched_targets = targets[within_tolerance]

        return matched_preds, matched_targets



def dict_to(x, device='cuda'):
    if isinstance(x, dict):
        return {k: x[k].to(device) for k in x}
    else:
        return x.to(device)

def to_device(x, device='cuda'):
    return tuple(dict_to(e,device) for e in x)

class DeviceDataLoader:
    def __init__(self, dataloader, device='cuda'):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            if isinstance(batch, list):
                yield [dict_to(x, self.device) for x in batch]
            else:
                yield dict_to(batch, self.device)



"""
def expand_patches(predictions, patch_size):
    # predictions shape: (batch_size, seq_len, num_features)

    # Expand the last dimension
    predictions = predictions.unsqueeze(-1)  # New shape: (batch_size, seq_len, num_features, 1)

    # Transpose to bring the feature dimension to the end
    predictions = predictions.permute(0, 1, 3, 2)  # New shape: (batch_size, seq_len, 1, num_features)

    # Tile the predictions to match the patch size
    predictions = predictions.repeat(1, 1, patch_size, 1)  # New shape: (batch_size, seq_len, patch_size, num_features)

    # Reshape to get the final expanded shape
    predictions = predictions.view(predictions.shape[0], predictions.shape[1] * predictions.shape[2], predictions.shape[3])
    # New shape: (batch_size, seq_len * patch_size, num_features)

    return predictions
"""

def expand_patches(predictions, patch_size):
    # predictions shape: (batch_size, seq_len, num_features)

    # Expand the last dimension
    predictions = predictions.unsqueeze(-1)  # New shape: (batch_size, seq_len, num_features, 1)

    # Transpose to bring the feature dimension to the end
    predictions = predictions.permute(0, 1, 3, 2)  # New shape: (batch_size, seq_len, 1, num_features)

    # Tile the predictions to match the patch size
    predictions = predictions.repeat(1, 1, patch_size, 1)  # New shape: (batch_size, seq_len, patch_size, num_features)

    # Reshape to get the final expanded shape
    predictions = predictions.view(predictions.shape[0], predictions.shape[1] * predictions.shape[2], predictions.shape[3])
    # New shape: (batch_size, seq_len * patch_size, num_features)

    return predictions



"""
class CustomLoss(nn.Module):
    def __init__(self, CFG, z_loss_weight=0.01):
        super(CustomLoss, self).__init__()
        self.ce = nn.BCEWithLogitsLoss(reduction='mean')
        self.z_loss_weight = z_loss_weight
        self.CFG = CFG

    def forward(self, inputs, targets, z=None):
        targets = targets['Y']

        # Expand patches for inputs and targets
        expanded_inputs = expand_patches(inputs, self.CFG['patch_size'])
        expanded_targets = expand_patches(targets, self.CFG['patch_size'])

        # Calculate the Binary Cross-Entropy loss
        bce_loss = self.ce(expanded_inputs, expanded_targets)

        # Calculate the z-loss if z is provided
        z_loss = 0.0
        if z is not None:
            z_loss = torch.mean(torch.abs(z))

        # Combine BCE loss with z-loss
        total_loss = bce_loss + self.z_loss_weight * z_loss

        return total_loss
"""


class CustomLoss(nn.Module):
    def __init__(self, CFG, z_loss_weight=0.01):
        super(CustomLoss, self).__init__()
        self.ce = nn.BCEWithLogitsLoss(reduction='none')
        self.z_loss_weight = z_loss_weight
        self.CFG = CFG

    def forward(self, inputs, targets, z=None):
        targets = targets['Y']
        
        # Expand patches for inputs and targets
        expanded_inputs = expand_patches(inputs, self.CFG['patch_size'])
        expanded_targets = expand_patches(targets, self.CFG['patch_size'])

        #expanded_inputs = inputs
        #expanded_targets = targets

        # Flatten the inputs and targets to apply BCE loss
        inputs_flat = expanded_inputs.view(-1)  # Flatten the inputs
        targets_flat = expanded_targets.view(-1)  # Flatten the targets
        
        # Calculate the Binary Cross-Entropy loss for each element
        bce_loss = self.ce(inputs_flat, targets_flat)
        
        # Sum the losses and divide by the total number of elements to get the mean loss
        total_loss = torch.sum(bce_loss) / inputs.numel()

        """Z = expanded_inputs.exp().sum(dim=-1).log()
        z_loss = (Z**2).mean() * self.z_loss_weight
        total_loss = total_loss + z_loss"""

        return total_loss




from fastai.metrics import Metric
from sklearn.metrics import average_precision_score
import numpy as np

class MAPMetric(Metric):
    "Mean Average Precision (MAP) for aligned prediction and target sequences"
    def __init__(self):
        self.reset()

    def reset(self):
        self.preds = []
        self.targs = []

    def accumulate(self, learn):
        preds = learn.pred
        targs = learn.yb[0]['Y']  # Assuming the targets are in the first element of the tuple
        # You might need to adjust this part if preds and targs need any specific preparation
        self.preds.append(preds)
        self.targs.append(targs)

    @property
    def value(self):
        all_preds = torch.cat(self.preds).cpu().numpy()
        all_targs = torch.cat(self.targs).cpu().numpy()

        # Ensure binary format by thresholding at 0.5
        all_targs_binary = (all_targs > 0.8).astype(int)

        ap_scores = []
        for i in range(all_preds.shape[1]):  # Iterate over each class
            ap = average_precision_score(all_targs_binary[:, i], all_preds[:, i])
            ap_scores.append(ap)

        return np.mean(ap_scores)


def push_to_github(file_path, commit_message):
    try:
        # Add the file to the local repository
        subprocess.run(["git", "add", file_path], check=True)
        
        # Commit the change
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        
        # Push the commit to the remote repository
        subprocess.run(["git", "push"], check=True)
        print(f"Successfully pushed {file_path} to GitHub.")
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to push {file_path} to GitHub.")
        print(str(e))