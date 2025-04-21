from collections import Counter

import anndata
import anndata as ad
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login, hf_hub_download
from spotd.spotd_gnn_vis import spotdf
from spotd.data_prepare import *
from PIL import Image
import cv2
import copy
from sklearn.model_selection import KFold

login("XXXXX") # input your personal token
model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))


class roi_dataset(Dataset):
    def __init__(self, img,
                 ):
        super().__init__()
        self.transform = transform
        self.images_lst = img

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        pil_image = Image.fromarray(self.images_lst[idx].astype('uint8'))
        image = self.transform(pil_image)
        return image
    
def crop_image(img, x, y, crop_size):
    # 计算子图左上角的坐标 (左上角为原点)
    left = x - crop_size[0] // 2
    top = y - crop_size[1] // 2

    # 计算子图右下角的坐标
    right = left + crop_size[0]
    bottom = top + crop_size[1]

    # 检查边界
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, img.shape[1])
    bottom = min(bottom, img.shape[0])

    cropped_img = img[top:bottom, left:right]
    return cropped_img, (left + crop_size[0] // 2, top + crop_size[1] // 2)

def extract_features(adata, model, image_path, batch_size=1):
    img = cv2.imread(image_path)
    img = np.array(img)

    spot_num = len(adata.obsm['spatial'])
    loc = adata.obsm['spatial']
    print(spot_num, loc)

    if isinstance(loc, pd.DataFrame):
        loc = loc.values

    output_folder = '../T1/Breast Cancer/file_tmp/spot_imgs_test100/'
    os.makedirs(output_folder, exist_ok=True)

    imagess = []
    for i in range(spot_num):
        x = loc[i, 0]
        y = loc[i, 1]
        image, center = crop_image(img, x, y, crop_size=(100, 100))
        sub_image_path = os.path.join(output_folder, f'sub_image_{i}_50x50.png')
        cv2.imwrite(sub_image_path, image)
        imagess.append(image)
    model.eval().cuda()
    feature_embs = []
    
    images = np.array(imagess)
    test_data = roi_dataset(images)
    database_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False)
    with torch.no_grad():
        for batch in database_loader:
            batch = batch.cuda()
            feature_emb = model(batch)
            feature_embs.append(feature_emb.cpu().numpy())

    feature_embs = np.concatenate(feature_embs, axis=0)
    return feature_embs

def main(a,b,cell_key):
    for i in range(a, b):
        st_file = 'X/Simulated_datasets/dataset' + str(i) + '/Spatial.h5ad'
        sc_file = 'X/Simulated_datasets/dataset' + str(i) + '/scRNA.h5ad'
        st_template_file = sc.read_visium(path='D:/Codings/Data/Data/1.DLPFC/151673', count_file='filtered_feature_bc_matrix.h5',
                                          library_id="mouse", load_images=True, source_image_path="/spatial/")
        sc_template_file = sc.read_h5ad('D:/Codings/Data/Data/1.DLPFC/151673/scRNA.h5ad')
        sc_data=copy.deepcopy(sc_template_file)
        st_data=copy.deepcopy(st_template_file)

        outfile = 'spotd\Result\dataset_dlf_Trans_' + str(i)
        datafile='Datasets/preproced_data\dataset_dlf' + str(i)
        if not os.path.exists(outfile):
            os.makedirs(outfile)
        if not os.path.exists(datafile):
            os.makedirs(datafile)

        # for simulated datasets
        # density = st_data1.uns['density'] 
        # cell_counts = st_data1.obs['cell_counts'].values 
        # cell_proportions = density.div(cell_counts,axis=0)#或者density / cell_counts[:, np.newaxis]

        """数据处理"""
        sc_adata = anndata.read_h5ad(datafile + '\Sm_STdata_filter.h5ad')
        st_adata = anndata.read_h5ad(datafile + '\Real_STdata_filter.h5ad')
        real_sc_adata = anndata.read_h5ad(datafile + '\Scdata_filter.h5ad')
        sm_labeled = anndata.read_h5ad(datafile + '\Sm_STdata_filter.h5ad')
        sm_data = pd.DataFrame(data=sc_adata.X.toarray(), columns=sc_adata.var_names)
        st_data = pd.DataFrame(data=st_adata.X.toarray(), columns=st_adata.var_names)
        # st_coords = st_adata.obsm['spatial']
        # st_label = st_adata.obs['ground_truth_encoded']

        sm_label = sm_labeled.obsm['label']
        # get image features
        # features = extract_features(st_adata, model, "D:/Codings/Data/BreastCancer1/V1_Breast_Cancer_Block_A_Section_1_image.tif", batch_size=1)
        # st_data = pd.DataFrame(data=st_adata.X, columns=st_adata.var_names)
        count_ct_dict = Counter(list(real_sc_adata.obs[cell_key]))
        celltypenum = len(count_ct_dict)
        # spottypenum = num_categories
        # st_label = st_adata.obs['ground_truth_encoded']
        print("------Start Running Stage------")
        model_da = MDCD(celltypenum, outdirfile=outfile, used_features=list(sm_data.columns), num_epochs=20)
        model_da.double_train(sm_data=sm_data, st_data=st_data, sm_label=sm_label)
        final_preds_target, final_predictions = model_da.prediction_overall_and_ratio()
        final_preds_target.columns = sm_label.columns.tolist()
        pd.DataFrame(data=final_preds_target).to_csv(outfile + '/final_pro.csv')
        final_predictions = pd.DataFrame(final_predictions, columns=st_adata.var_names)
        final_predictions.to_csv(outfile + '/final_overall.csv')