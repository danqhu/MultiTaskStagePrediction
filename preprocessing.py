from utils import feature_preprocessing, imputate_missing_values, one_hot_encoding, feature_normalization
import pandas as pd
import numpy as np
import copy



label_flag = 'N2'
petct_subset = False




data = pd.read_csv('~/data/projects_dataset/SimilarPatientNetworkData/data/multimodel_data_8.csv')                   # 读取标注数据
data.dropna(subset=["Stage_pN",  "2"], inplace=True)          # 剔除出没有病理分期和增强CT影像的患者 , "SUV_zongge",



for x in data.index:
    if data.loc[x, "Stage_pN"] == "x":                  # 剔除病理分期为x的项
        data.drop(x, inplace=True)


# temp = data.duplicated(subset=['sys_patientNo'])




'''
prediction labels
'''
labels_pN = data["Stage_pN"].astype("int32")
labels_binary = copy.deepcopy(labels_pN)
labels_binary[labels_pN == 0] = 0
labels_binary[labels_pN == 1] = 0 if label_flag == 'N2' else 1
labels_binary[labels_pN == 2] = 1
labels_binary[labels_pN == 3] = 1


# print(labels_binary.value_counts())

labels_binary.to_csv('~/data/projects_dataset/SimilarPatientNetworkData/data/labels_binary_' + label_flag + '.csv',index=False)





'''
Thoracic Surgon Evaluation
'''
labels_surgon = data["Stage_cN"].to_numpy().astype(np.int)
labels_surgon[labels_surgon==0] = 0
labels_surgon[labels_surgon==1] = 0 if label_flag == 'N2' else 1
labels_surgon[labels_surgon==2] = 1
labels_surgon[labels_surgon==3] = 1



# '''
# PET/CT SUVmax
# '''
# data_petct = copy.deepcopy(data["SUV_zongge_reviewed"].to_numpy().astype(np.float))
# data_petct[np.isnan(data_petct)] = 0.0
# data_petct[data_petct <= 2.5] = 0
# data_petct[data_petct > 2.5] = 1
# data_petct = data_petct.astype(np.int)




'''
CT short length
'''
labels_ct = copy.deepcopy(data["纵隔淋巴结短径"].to_numpy().astype(np.float))
labels_ct[np.isnan(labels_ct)] = 0.0
labels_ct[labels_ct <= 1.0] = 0
labels_ct[labels_ct > 1.0] = 1
labels_ct = labels_ct.astype(np.int)


# if label_flag == 'N1':
#     data_ct_hilar = copy.deepcopy(data["肺门淋巴结短径_reviewed"].to_numpy().astype(np.float))
#     data_ct_hilar[np.isnan(data_ct_hilar)] = 0.0
#     data_ct_hilar[data_ct_hilar <= 1.0] = 0
#     data_ct_hilar[data_ct_hilar > 1.0] = 1
#     data_ct_hilar = data_ct_hilar.astype(np.int)
#     data_ct[data_ct == 0] = data_ct_hilar[data_ct == 0]



print(np.bincount(labels_ct))




data_col_name_selected = [

    "主肿物位置",
    "主肿物长径",
    "主肿物短径",
    "主肿物形状_毛糙",
    "主肿物形状_分叶",
    "主肿物磨玻璃或实性",
    "是否侵犯血管",
    "是否侵犯胸膜",
    # "是否侵犯支气管_reviewed",
    "肺门淋巴结长径",
    "肺门淋巴结短径",
    "纵隔淋巴结长径",
    "纵隔淋巴结短径",
    "Age",
    "Smoking_history",
    "Drinking_history",
    "Family_tumor_history",
    "Gender",
    "高血压",
    "糖尿病",
    "肺结核",
    "心血管疾病",
    "脑血管疾病",
    "Pretreatment_CEA1",
    "Pretreatment_CA1991",
    "Pretreatment_CA1251",
    "Pretreatment_NSE1",
    "Pretreatment_CYFRA2111",
    "Pretreatment_SCC1",
]


# deepfeature from CT images
col_deepfeature = [str(x) for x in range(2,10)]
data_col_name_selected.extend(col_deepfeature)


if petct_subset:
    col_petct = ['Pretreatment_PETCT_tumor_SUVmax', 'SUV_zongge', 'SUV_feimen']
    data_col_name_selected.extend(col_petct)


data_selected = data[data_col_name_selected]


'''检查是否所有的临床数据都有对应的影像数据'''
# data_image_filename = data[['Image_id', 'CT_exam_id2']].astype({'Image_id':'int32', 'CT_exam_id2':'int32'})
# data_image_file_path = './CT_images/'
# files = os.listdir(data_image_file_path)
# for index, row in data_image_filename.iterrows():
#     name = str(row['Image_id'])+'_'+str(row['CT_exam_id2'])+'.pt'
#     if name not in files:
#         print(name)


data_selected = feature_preprocessing(data=data_selected, binarize_size=True)
data_selected = imputate_missing_values(data_selected, binarize_size=True)


data_selected_normalized = feature_normalization(data_selected, normalization_type="normalization")
data_selected_standardlized = feature_normalization(data_selected, normalization_type="standardlization")


data_selected.reset_index(inplace=True, drop=True)
data_selected_normalized.reset_index(inplace=True, drop=True)
data_selected_standardlized.reset_index(inplace=True, drop=True)



data_one_hot = one_hot_encoding(data_selected, drop_first=False)
data_one_hot_normalized = one_hot_encoding(data_selected_normalized, drop_first=False)
data_one_hot_standardlized = one_hot_encoding(data_selected_standardlized, drop_first=False)
features_list = data_one_hot.columns.to_numpy()


'''保存预处理后的数据'''
# data_one_hot.to_csv('~/data/projects_dataset/SimilarPatientNetworkData/data/data_one_hot.csv',index=False)
# data_one_hot_normalized.to_csv('~/data/projects_dataset/SimilarPatientNetworkData/data/data_one_hot_normalized.csv',index=False)
# data_one_hot_standardlized.to_csv('~/data/projects_dataset/SimilarPatientNetworkData/data/data_one_hot_standardlized.csv',index=False)

'''保存预处理后的数据对应的影像编号'''
data_image_filename = data[['Image_id', 'CT_exam_id2']].astype({'Image_id':'int32', 'CT_exam_id2':'int32'})
data_image_filename.to_csv('~/data/projects_dataset/SimilarPatientNetworkData/data/data_image_filename.csv',index=False)


