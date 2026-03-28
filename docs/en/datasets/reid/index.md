---
comments: true
description: Learn how to structure datasets for YOLO ReID tasks. Detailed folder structure, naming conventions, and usage examples for person re-identification training.
keywords: YOLO, person re-identification, ReID, dataset structure, Market-1501, Ultralytics, metric learning, training data
---

# ReID Datasets Overview

## Dataset Structure for YOLO ReID Tasks

For [Ultralytics](https://www.ultralytics.com/) YOLO re-identification tasks, the dataset must follow a specific structure where person images are organized under a root directory with separate splits for training, query, and gallery sets.

### Image Naming Convention

ReID datasets use a filename convention that encodes person identity (PID) and camera ID (CamID):

```
PPPP_cCsSXXX_XXXXXX.jpg
```

- `PPPP`: Person ID (e.g., `0001`, `0002`)
- `C`: Camera ID (e.g., `1` through `6`)
- `S`: Sequence number
- `XXXXXX`: Frame number

For example, `0001_c1s1_001051_00.jpg` means person `0001` captured by camera `1`.

### Folder Structure Example

Consider the [Market-1501](market1501.md) dataset as an example. The folder structure should look like this:

```
Market-1501-v15.09.15/
|
|-- bounding_box_train/
|   |-- 0001_c1s1_001051_00.jpg
|   |-- 0001_c1s1_009376_00.jpg
|   |-- 0001_c2s1_001051_00.jpg
|   |-- 0002_c1s1_001051_00.jpg
|   |-- ...
|
|-- query/
|   |-- 0001_c1s1_001051_00.jpg
|   |-- 0002_c4s2_012428_00.jpg
|   |-- ...
|
|-- bounding_box_test/
|   |-- 0001_c1s1_001051_00.jpg
|   |-- 0001_c2s2_065432_00.jpg
|   |-- ...
```

### Dataset YAML Format

A ReID dataset YAML config specifies the root path, split directories, and number of training identities:

```yaml
path: Market-1501-v15.09.15  # dataset root dir
train: bounding_box_train     # training images
val: query                    # query images for evaluation
gallery: bounding_box_test    # gallery images for evaluation

nc: 751  # number of training identities
```

!!! note

    Unlike detection or classification datasets, ReID datasets require a `gallery` field that specifies the gallery set used during evaluation. The evaluation protocol compares each query image against all gallery images to compute mAP and Rank-1 metrics.

## Usage

To train a YOLO ReID model on a dataset, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-reid.yaml")

        # Train the model
        results = model.train(data="Market-1501.yaml", epochs=60, imgsz=256)
        ```

    === "CLI"

        ```bash
        # Start training from a YAML model config
        yolo reid train data=Market-1501.yaml model=yolo26n-reid.yaml epochs=60 imgsz=256
        ```

## Supported Datasets

- [Market-1501](market1501.md) - The most widely used ReID benchmark with 751 training identities from 6 cameras.

## FAQ

### What is the evaluation protocol for ReID datasets?

ReID evaluation follows a query-gallery protocol. Each query image is compared against all gallery images by computing embedding distances. For each query, gallery images are ranked by distance, and standard retrieval metrics are computed:

- **mAP (mean Average Precision)**: Measures overall retrieval quality across all queries.
- **Rank-1**: The fraction of queries where the correct identity appears as the top-ranked result.

Same-person-same-camera matches are excluded from evaluation, as they are trivially easy to match.

### How does ReID dataset structure differ from classification datasets?

Classification datasets organize images into class subdirectories (e.g., `cat/`, `dog/`). ReID datasets instead encode identity information in the filename (e.g., `0001_c1s1_001051_00.jpg`) and keep all images in a flat directory. This is because ReID also needs camera ID information for the evaluation protocol.

### Can I use custom ReID datasets with YOLO?

Yes. Create a YAML config file with `path`, `train`, `val`, `gallery`, and `nc` fields pointing to your dataset. Images must follow the `PPPP_cCsSXXX_XXXXXX.jpg` naming convention so that person IDs and camera IDs can be parsed.
