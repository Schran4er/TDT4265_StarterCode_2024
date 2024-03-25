take task2 !

Cybele lab hours:
Mondays 15:15 - 17:00 and
Thursdays 12:00 - 14:00


Plan:
 - Data exploration: nrrd, vtp, stl data
        Diseased/Normal
        ├── Annotations --> images segmentated by the 3 experts manually -> ground truth!
        ├── CTCA --> whole scans (2D) / raw data! goal: detect aorta, generate 3d (centerline & surfacemeshes), classify normal/diseased
        ├── Centerlines --> centerlines of aortas
        ├── SurfaceMeshes --> aorta surfaces
        └── Testset_Disease/Testset_Normal -> useless because not labelled

- see MONAI

- Pre-processing pipeline:
    -> divide into training/test/val (because testset doesn't have labels)
    -> Data augmentation: crop/rotate/flip/... (look at previous task (assignement 3))
    - Convert data to usable format for a ML model:
        - use 3D augmentation (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9952534/ "Comparing 3D, 2.5D, and 2D Approaches to Brain Image Auto-Segmentation")

    - image contrast normalization to zero mean and unit variance (ASOCA paper submission 2)
    - Frangi/... filters not/? used (as ASOCA paper states that they are too computationally expensive)


- build baseline model
    - Look for pre trained model for medical image segmentation (from literature) which has not been trained on the asoca dataset!!
- build training setup!!
    - loss function: Soft Dice loss + Cross entropy loss or focal loss (tensorboard for keeping track of the loss?)
    - performance metrics
    - optimizer
    - regularization
    - early stopping
    - pre training (take pre trained model, adjust it, train on our data)
    - speed up training with batch normalization

- post-processsing: remove small disconnected components

- do improvements
    - note and justify/explain the import changes!!
- do the same with another architecture


- runtime analysis
    - runtime = inference time: how fast can the model detect? (in real application)
- carbon footprint analysis
 - total training time * GPU_power = work -> convert to how far we can drive with a tesla


Documentation:
- Readme for code (how to train, use the model etc., conda env)
- Technical documentation
- Presentation


Further work:
- other filters (check also other submissions from ASOCA paper)
- further post-processing magic (check also other submissions from ASOCA paper)