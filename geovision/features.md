#### TODO:
Priority Features
Use Autoreload in IPython

Write Datasets for EuroSAT, Resisc, Urban Footprint
Write Datasets for So2Sat-LCZ, So2Sat-Pop, Sen12Floods, BigEarthNet-MM, PASTIS-R
Write Dataset using TorchGeo for Hypercrop (understand how it geo-samples and reprojects at runtime)


#### TODO:
Important Features

IMPORTANT: CLEANUP THE RUNTIME ELEMENTS 
           Supress lightning debugging info (add a verbose: bool option?), 
           Only have one progress bar at a time and remove it when done 

Classification Report: Confusion Matrix -> Classwise + Macro + Micro Metrics
For Classification Tasks: Log filenames + labels which were incorrectly classified from the validation dataset
For Segmentation Tasks: Log filenames + iou + dice which were below a threshold from the validation dataset
Visualize training loss and validation loss during training using matplotlib animations 

Learninng Rate Scheduler (after some experience with training large models)

#### TODO:
QoL Features 
Experiment Config using Pydantic to Validate Parameters  
Added Status Messages to DataModule to reflect experimental setup (peace of mind)

Make Checkpointing and Logging more elegant -> Think / Read Other Solutions
Display Checkpoint: .ckpt file -> Visualize Model, Experimental Config, Training History, etc
Display Metrics: .csv log file -> Visualize Training Metrics (Value vs Epoch), Confusion Matrix (wandb style), etc.
Figure out how to log negative samples (misclassified samples): Modified CSV Logger perhaps?

#### TODO:
QoL Artifacts:

To evaluate a .ckpt (1 epoch) on a eval dataset
1. Confusion Matrix: Classification Report
2. Negative Samples DataFrame: Grid Picture(a few samples / all samples) + Metric(s) + Attribution Map
3. Training History: Visualize Loss + Monitored Metrics vs Epoch / Step (read torchmetrics plotting)


#### TODO:
During training
Animated Train Loss (+ Val Loss) vs Epoch / Step 
Animated Monitored Metric vs Epoch / Step

#### TODO:
Experimental Parameters (as general as possible):
Parameters
__________
dataset_name: name of the dataset, used to name the root directory 
task: Literal["classification", "segmentation"], used to name bucket on the cloud
random_seed: for deterministic experments, mandatory