from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.pyplot import rc_context
from PIL import Image
from tqdm import tqdm

from batchdetect.batchdetect import BatchDetect
from batchdetect.image import (conch, ctranspath, first_and_second_order,
                               h0_mini, h_optimus_0, h_optimus_1, resnet, uni,
                               uni2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets = [
    'LungCancer',
    'CRC', 
]

models = [
    "conch",
    "uni",
    "first_and_second_order", 
    "resnet", 
    "ctranspath",
    "h_optimus_0", 
    "h0_mini", 
    "h_optimus_1", 
    "uni2",
]


for dataset in datasets:
    method = "original"  # no batch correction method applied

    # create metadata dataframe from clini_table and folder structure
    base_dir = Path(f'/lustre/groups/shared/users/peng_marr/BatchDetect/BatchDetect{dataset}')
    clini_table = pd.read_csv(base_dir / f'BatchDetect{dataset}_clini.csv')

    labels = list(clini_table.columns)  # or costum list
    labels.remove('PATIENT')
    labels.remove('AGE')
    if 'sample_id' in labels:
        labels.remove('sample_id')

    metadata_path = Path(base_dir / 'metadata.csv')
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
    else:
        # metadata with columns: file, label (MSI-H), submission site
        patch_list = list(base_dir.glob('**/*.jpeg'))
        print('Number of patches:', len(patch_list))

        submission_site = [patch.parent.parent.name for patch in patch_list]
        metadata = pd.DataFrame(list(zip(patch_list, submission_site)), columns=['file', 'dataset'])

        for l in labels:
            if dataset == 'CRC':
                label = [clini_table[l][clini_table['PATIENT'] == patch.name.split('_')[0]].item() for patch in patch_list]
            else:
                label = [clini_table[l][clini_table['PATIENT'] == patch.parent.name[:12]].item() for patch in patch_list]
            metadata[l] = label
        metadata.to_csv(metadata_path, index=False)
        
    if dataset == 'CRC':
        # for TCGA-CRC cohorts
        from pathlib import Path
        metadata["type"] = metadata["file"].astype(str).apply(lambda x: Path(x).parent.name.split(".")[0].split("-")[-1])
        metadata["type"] = metadata["type"].apply(lambda x: "FFPE" if x.startswith("DX") else x)
        metadata["type"] = metadata["type"].apply(lambda x: "frozen" if x.startswith("TS") else x)
        metadata["type"] = metadata["type"].apply(lambda x: "frozen" if x.startswith("BS") else x)
        # map type to frozen if dataset == CPTAC
        metadata["type"] = metadata.apply(lambda x: "frozen" if x["dataset"] == "CPTAC" else x["type"], axis=1)
        # map all entries that are not frozen or FFPE to ""
        metadata["type"] = metadata["type"].apply(lambda x: x if x in ["frozen", "FFPE"] else np.nan)
        # metadata.to_csv(Path(base_dir / 'metadata.csv'), index=False)
        labels = labels + ["type"]
        
    for features in tqdm(models):
        try: 
            df_features_path = base_dir / f'{method}_{features}_features.csv'

            if df_features_path.exists():
                df_features = pd.read_csv(df_features_path)
            else:
                if features == 'first_and_second_order':
                    df_features = first_and_second_order(metadata)
                elif features == 'resnet':
                    df_features = resnet(metadata)
                    df_features = pd.DataFrame(np.stack(df_features, axis=0))
                elif features == 'ctranspath':
                    df_features = ctranspath(metadata)
                    df_features = pd.DataFrame(np.stack(df_features, axis=0))
                elif features == 'h_optimus_0':
                    df_features = h_optimus_0(metadata)
                    df_features = pd.DataFrame(np.stack(df_features, axis=0))
                elif features == 'h0_mini':
                    df_features = h0_mini(metadata)
                    df_features = pd.DataFrame(np.stack(df_features, axis=0)[:, 0, :])
                elif features == 'h_optimus_1':
                    df_features = h_optimus_1(metadata)
                    df_features = pd.DataFrame(np.stack(df_features, axis=0))
                elif features == 'uni2':
                    df_features = uni2(metadata)
                    df_features = pd.DataFrame(np.stack(df_features, axis=0))
                elif features == 'uni':
                    df_features = uni(metadata)
                    df_features = pd.DataFrame(np.stack(df_features, axis=0))
                elif features == 'conch': 
                    # custom forward pass
                    model, transform = conch()
                    file_list = metadata["file"].tolist()
                    df_features = []
                    for f in tqdm(file_list):
                        image = Image.open(f).convert('RGB')

                        with torch.no_grad():
                            with torch.cuda.amp.autocast():
                                image = transform(image).unsqueeze(0)
                                image = image.to(device)
                                df_features.append(model.encode_image(image, proj_contrast=False, normalize=False).squeeze().cpu().numpy())
                    df_features = pd.DataFrame(np.stack(df_features, axis=0))
                else:
                    raise ValueError(f"Unknown features: {features}")
                
                df_features.to_csv(df_features_path, index=False)
        
            bd = BatchDetect(metadata.loc[:, [*labels, "dataset"]], df_features)

            with rc_context({"figure.figsize": (4, 4)}):
                ax = bd.low_dim_visualization("umap")  # Prevent immediate display
                plt.savefig(f'./figures/umap_{dataset}_{features}.png', dpi=300, bbox_inches='tight', pad_inches=0)  # Save at 600 DPI
                plt.close()
            
        except Exception as e:
            print(f"Error in {dataset}, {features}: {e}")
            continue
