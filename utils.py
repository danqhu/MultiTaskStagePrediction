import pandas as pd
import copy
from scipy.stats import chi2_contingency, mannwhitneyu
import numpy as np
import pickle


def save_dict(dict, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(dict, f)


def load_dict(file_path):
    with open(file_path, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict

def feature_preprocessing(data: pd.DataFrame, binarize_size=False, binarize_SUVmax=False):
    data = copy.deepcopy(data)


    if 'Smoking_history' in data.columns:
        for idx, row in data.iterrows():
            if row['Smoking_history'] == '有':
                data.at[idx, 'Smoking_history'] = '1'
            elif row['Smoking_history'] == '无':
                data.at[idx, 'Smoking_history'] = '0'



    if 'Drinking_history' in data.columns:
        for idx, row in data.iterrows():
            if row['Drinking_history'] == '有':
                data.at[idx, 'Drinking_history'] = '1'
            elif row['Drinking_history'] == '无':
                data.at[idx, 'Drinking_history'] = '0'



    if 'Family_tumor_history' in data.columns:
        for idx, row in data.iterrows():
            if row['Family_tumor_history'] == '有':
                data.at[idx, 'Family_tumor_history'] = '1'
            elif row['Family_tumor_history'] == '无':
                data.at[idx, 'Family_tumor_history'] = '0'

    if 'Gender' in data.columns:
        for idx, row in data.iterrows():
            if row['Gender'] == '男':
                data.at[idx, 'Gender'] = '1'
            elif row['Gender'] == '女':
                data.at[idx, 'Gender'] = '0'




    if binarize_size:
        if '纵隔淋巴结长径' in data.columns:
            for idx, row in data.iterrows():
                if row['纵隔淋巴结长径'] > 1.0:
                    data.at[idx, '纵隔淋巴结长径'] = 1.0
                else:
                    data.at[idx, '纵隔淋巴结长径'] = 0

        if '纵隔淋巴结短径' in data.columns:
            for idx, row in data.iterrows():
                if row['纵隔淋巴结短径'] > 1.0:
                    data.at[idx, '纵隔淋巴结短径'] = 1.0
                else:
                    data.at[idx, '纵隔淋巴结短径'] = 0

        if '肺门淋巴结长径' in data.columns:
            for idx, row in data.iterrows():
                if row['肺门淋巴结长径'] > 1.0:
                    data.at[idx, '肺门淋巴结长径'] = 1.0
                else:
                    data.at[idx, '肺门淋巴结长径'] = 0

        if '肺门淋巴结短径' in data.columns:
            for idx, row in data.iterrows():
                if row['肺门淋巴结短径'] > 1.0:
                    data.at[idx, '肺门淋巴结短径'] = 1.0
                else:
                    data.at[idx, '肺门淋巴结短径'] = 0
        

    if binarize_SUVmax:
        if 'SUV_zongge' in data.columns:
            for idx, row in data.iterrows():
                if row['SUV_zongge'] > 2.5:
                    data.at[idx, 'SUV_zongge'] = 1.0
                else:
                    data.at[idx, 'SUV_zongge'] = 0

        if 'SUV_feimen' in data.columns:
            for idx, row in data.iterrows():
                if row['SUV_feimen'] > 2.5:
                    data.at[idx, 'SUV_feimen'] = 1.0
                else:
                    data.at[idx, 'SUV_feimen'] = 0

    



    if '主肿物位置' in data.columns:
        for idx, row in data.iterrows():
            if '左肺上叶' == row['主肿物位置'] or '左肺下叶' == row['主肿物位置'] or '右肺上叶' == row['主肿物位置'] or '右肺中叶' == row['主肿物位置'] or '右肺下叶' == row['主肿物位置']:
                continue


            if '左肺尖' in row['主肿物位置']:
                data.at[idx, '主肿物位置'] = '左肺上叶'
            elif '右肺尖' in row['主肿物位置']:
                data.at[idx, '主肿物位置'] = '右肺上叶'
            # elif '左' in row['主肿物位置'] and '肺门' in row['主肿物位置']:
            #     data.at[idx, '主肿物位置'] = '其他'
            # elif '右' in row['主肿物位置'] and '肺门' in row['主肿物位置']:
            #     data.at[idx, '主肿物位置'] = '其他'
            else:
                data.at[idx, '主肿物位置'] = '其他'

    

    if '主肿物磨玻璃或实性' in data.columns:
        data = data.astype({'主肿物磨玻璃或实性': 'int'})
        data = data.astype({'主肿物磨玻璃或实性': 'string'})
        for idx, row in data.iterrows():
            if row['主肿物磨玻璃或实性'] == '0':
                data.at[idx, '主肿物磨玻璃或实性'] = '实性'
            elif row['主肿物磨玻璃或实性'] == '1':
                data.at[idx, '主肿物磨玻璃或实性'] = '实性'
            elif row['主肿物磨玻璃或实性'] == '2':
                data.at[idx, '主肿物磨玻璃或实性'] = '磨玻璃'
            elif row['主肿物磨玻璃或实性'] == '3':
                data.at[idx, '主肿物磨玻璃或实性'] = '混杂磨玻璃'
            else:
                data.at[idx, '主肿物磨玻璃或实性'] = '其他'

    return data


def imputate_missing_values(data: pd.DataFrame, binarize_size=True, binarize_SUVmax=False):
    data = copy.deepcopy(data)
    for col_name in data.columns:

        if "Stage_cT" in col_name:
            mode_val = data[col_name].mode()[0]
            data[col_name].fillna(mode_val, inplace=True)

        if "Stage_cN" in col_name:
            mode_val = data[col_name].mode()[0]
            data[col_name].fillna(mode_val, inplace=True)


        if "主肿物位置" in col_name:
            mode_val = data[col_name].mode()[0]
            data[col_name].fillna(mode_val, inplace=True)


        if "主肿物长径" in col_name:
            mode_val = data[col_name].mode()[0]
            min_val = data[col_name].min()
            max_val = data[col_name].max()
            mean_val = data[col_name].mean()
            median_val = data[col_name].median()
            std_val = data[col_name].std()
            data[col_name].fillna(median_val, inplace=True)
            print(col_name + str(median_val))
            # if imputate_type == "normalization":
            #     data[col_name] = (data[col_name].to_numpy() - min_val)/(max_val - min_val)
            # elif imputate_type == "standardlization":
            #     data[col_name] = (data[col_name].to_numpy() - mean_val) / std_val


        if "主肿物短径" in col_name:
            mode_val = data[col_name].mode()[0]
            min_val = data[col_name].min()
            max_val = data[col_name].max()
            mean_val = data[col_name].mean()
            median_val = data[col_name].median()
            std_val = data[col_name].std()
            data[col_name].fillna(median_val, inplace=True)
            print(col_name + str(median_val))
            # if imputate_type == "normalization":
            #     data[col_name] = (data[col_name].to_numpy() - min_val) / (max_val - min_val)
            # elif imputate_type == "standardlization":
            #     data[col_name] = (data[col_name].to_numpy() - mean_val) / std_val

        if "主肿物形状_毛刺" in col_name:
            data[col_name].fillna(0, inplace=True)

        if "主肿物形状_毛糙" in col_name:
            data[col_name].fillna(0, inplace=True)

        if "主肿物形状_分叶" in col_name:
            data[col_name].fillna(0, inplace=True)

        if "主肿物磨玻璃或实性" in col_name:
            data[col_name].fillna(0, inplace=True)

        if "是否侵犯血管" in col_name:
            data[col_name].fillna(0, inplace=True)

        if "是否侵犯胸膜" in col_name:
            data[col_name].fillna(0, inplace=True)

        if "是否侵犯支气管" in col_name:
            data[col_name].fillna(0, inplace=True)

        if "肺门淋巴结长径" in col_name:
            mode_val = data[col_name].mode()[0]
            min_val = data[col_name].min()
            max_val = data[col_name].max()
            mean_val = data[col_name].mean()
            median_val = data[col_name].median()
            std_val = data[col_name].std()
            data[col_name].fillna(0, inplace=True)
            print(col_name + str(median_val))
            # if binarization:
            #     x = data[col_name].to_numpy()
            #     x[x >= 1.0] = 1
            #     x[x < 1.0] = 0
            #     data[col_name] = x
            # else:
            #     if imputate_type == "normalization":
            #         data[col_name] = (data[col_name].to_numpy() - min_val) / (max_val - min_val)
            #     elif imputate_type == "standardlization":
            #         data[col_name] = (data[col_name].to_numpy() - mean_val) / std_val

        if "肺门淋巴结短径" in col_name:
            mode_val = data[col_name].mode()[0]
            min_val = data[col_name].min()
            max_val = data[col_name].max()
            mean_val = data[col_name].mean()
            median_val = data[col_name].median()
            std_val = data[col_name].std()
            data[col_name].fillna(0, inplace=True)
            print(col_name + str(median_val))
            # if binarization:
            #     x = data[col_name].to_numpy()
            #     x[x >= 1.0] = 1
            #     x[x < 1.0] = 0
            #     data[col_name] = x
            # else:
            #     if imputate_type == "normalization":
            #         data[col_name] = (data[col_name].to_numpy() - min_val) / (max_val - min_val)
            #     elif imputate_type == "standardlization":
            #         data[col_name] = (data[col_name].to_numpy() - mean_val) / std_val

        if "纵隔淋巴结长径" in col_name:
            mode_val = data[col_name].mode()[0]
            min_val = data[col_name].min()
            max_val = data[col_name].max()
            mean_val = data[col_name].mean()
            median_val = data[col_name].median()
            std_val = data[col_name].std()
            data[col_name].fillna(0, inplace=True)
            print(col_name + str(median_val))
            # if binarization:
            #     x = data[col_name].to_numpy()
            #     x[x >= 1.0] = 1
            #     x[x < 1.0] = 0
            #     data[col_name] = x
            # else:
            #     if imputate_type == "normalization":
            #         data[col_name] = (data[col_name].to_numpy() - min_val) / (max_val - min_val)
            #     elif imputate_type == "standardlization":
            #         data[col_name] = (data[col_name].to_numpy() - mean_val) / std_val

        if "纵隔淋巴结短径" in col_name:
            mode_val = data[col_name].mode()[0]
            min_val = data[col_name].min()
            max_val = data[col_name].max()
            mean_val = data[col_name].mean()
            median_val = data[col_name].median()
            std_val = data[col_name].std()
            data[col_name].fillna(0, inplace=True)
            print(col_name + str(median_val))
            # if binarization:
            #     x = data[col_name].to_numpy()
            #     x[x >= 1.0] = 1
            #     x[x < 1.0] = 0
            #     data[col_name] = x
            # else:
            #     if imputate_type == "normalization":
            #         data[col_name] = (data[col_name].to_numpy() - min_val) / (max_val - min_val)
            #     elif imputate_type == "standardlization":
            #         data[col_name] = (data[col_name].to_numpy() - mean_val) / std_val

        if "Age" in col_name:
            mode_val = data[col_name].mode()[0]
            min_val = data[col_name].min()
            max_val = data[col_name].max()
            mean_val = data[col_name].mean()
            median_val = data[col_name].median()
            std_val = data[col_name].std()
            data[col_name].fillna(median_val, inplace=True)
            # if imputate_type == "normalization":
            #     data[col_name] = (data[col_name].to_numpy() - min_val) / (max_val - min_val)
            # elif imputate_type == "standardlization":
            #     data[col_name] = (data[col_name].to_numpy() - mean_val) / std_val

        if "Smoking_history" in col_name:
            mode_val = data[col_name].mode()[0]
            data[col_name].fillna(mode_val, inplace=True)

        if "Drinking_history" in col_name:
            mode_val = data[col_name].mode()[0]
            data[col_name].fillna(mode_val, inplace=True)

        if "Family_tumor_history" in col_name:
            mode_val = data[col_name].mode()[0]
            data[col_name].fillna(mode_val, inplace=True)

        if "Gender" in col_name:
            mode_val = data[col_name].mode()[0]
            data[col_name].fillna(mode_val, inplace=True)

        if "高血压" in col_name:
            mode_val = data[col_name].mode()[0]
            data[col_name].fillna(mode_val, inplace=True)

        if "糖尿病" in col_name:
            mode_val = data[col_name].mode()[0]
            data[col_name].fillna(mode_val, inplace=True)

        if "肺结核" in col_name:
            mode_val = data[col_name].mode()[0]
            data[col_name].fillna(mode_val, inplace=True)

        if "心血管疾病" in col_name:
            mode_val = data[col_name].mode()[0]
            data[col_name].fillna(mode_val, inplace=True)

        if "脑血管疾病" in col_name:
            mode_val = data[col_name].mode()[0]
            data[col_name].fillna(mode_val, inplace=True)

        if "Pretreatment_CEA1" in col_name:
            mode_val = data[col_name].mode()[0]
            min_val = data[col_name].min()
            max_val = data[col_name].max()
            mean_val = data[col_name].mean()
            median_val = data[col_name].median()
            std_val = data[col_name].std()
            data[col_name].fillna(median_val, inplace=True)
            print(col_name + str(median_val))
            # if imputate_type == "normalization":
            #     data[col_name] = (data[col_name].to_numpy() - min_val) / (max_val - min_val)
            # elif imputate_type == "standardlization":
            #     data[col_name] = (data[col_name].to_numpy() - mean_val) / std_val

        if "Pretreatment_CA1991" in col_name:
            mode_val = data[col_name].mode()[0]
            min_val = data[col_name].min()
            max_val = data[col_name].max()
            mean_val = data[col_name].mean()
            median_val = data[col_name].median()
            std_val = data[col_name].std()
            data[col_name].fillna(median_val, inplace=True)
            print(col_name + str(median_val))
            # if imputate_type == "normalization":
            #     data[col_name] = (data[col_name].to_numpy() - min_val) / (max_val - min_val)
            # elif imputate_type == "standardlization":
            #     data[col_name] = (data[col_name].to_numpy() - mean_val) / std_val

        if "Pretreatment_CA1251" in col_name:
            mode_val = data[col_name].mode()[0]
            min_val = data[col_name].min()
            max_val = data[col_name].max()
            mean_val = data[col_name].mean()
            median_val = data[col_name].median()
            std_val = data[col_name].std()
            data[col_name].fillna(median_val, inplace=True)
            print(col_name + str(median_val))
            # if imputate_type == "normalization":
            #     data[col_name] = (data[col_name].to_numpy() - min_val) / (max_val - min_val)
            # elif imputate_type == "standardlization":
            #     data[col_name] = (data[col_name].to_numpy() - mean_val) / std_val


        if "Pretreatment_NSE1" in col_name:
            mode_val = data[col_name].mode()[0]
            min_val = data[col_name].min()
            max_val = data[col_name].max()
            mean_val = data[col_name].mean()
            median_val = data[col_name].median()
            std_val = data[col_name].std()
            data[col_name].fillna(median_val, inplace=True)
            print(col_name + str(median_val))
            # if imputate_type == "normalization":
            #     data[col_name] = (data[col_name].to_numpy() - min_val) / (max_val - min_val)
            # elif imputate_type == "standardlization":
            #     data[col_name] = (data[col_name].to_numpy() - mean_val) / std_val

        if "Pretreatment_CYFRA2111" in col_name:
            mode_val = data[col_name].mode()[0]
            min_val = data[col_name].min()
            max_val = data[col_name].max()
            mean_val = data[col_name].mean()
            median_val = data[col_name].median()
            std_val = data[col_name].std()
            data[col_name].fillna(median_val, inplace=True)
            print(col_name + str(median_val))
            # if imputate_type == "normalization":
            #     data[col_name] = (data[col_name].to_numpy() - min_val) / (max_val - min_val)
            # elif imputate_type == "standardlization":
            #     data[col_name] = (data[col_name].to_numpy() - mean_val) / std_val


        if "Pretreatment_SCC1" in col_name:
            mode_val = data[col_name].mode()[0]
            min_val = data[col_name].min()
            max_val = data[col_name].max()
            mean_val = data[col_name].mean()
            median_val = data[col_name].median()
            std_val = data[col_name].std()
            data[col_name].fillna(median_val, inplace=True)
            print(col_name + str(median_val))
            # if imputate_type == "normalization":
            #     data[col_name] = (data[col_name].to_numpy() - min_val) / (max_val - min_val)
            # elif imputate_type == "standardlization":
            #     data[col_name] = (data[col_name].to_numpy() - mean_val) / std_val


        if "Pretreatment_PETCT_tumor_SUVmax" in col_name:
            mode_val = data[col_name].mode()[0]
            min_val = data[col_name].min()
            max_val = data[col_name].max()
            mean_val = data[col_name].mean()
            median_val = data[col_name].median()
            std_val = data[col_name].std()
            data[col_name].fillna(0, inplace=True)
            print(col_name + str(median_val))
            # if imputate_type == "normalization":
            #     data[col_name] = (data[col_name].to_numpy() - min_val) / (max_val - min_val)
            # elif imputate_type == "standardlization":
            #     data[col_name] = (data[col_name].to_numpy() - mean_val) / std_val

        if "SUV_zongge" in col_name:
            mode_val = data[col_name].mode()[0]
            min_val = data[col_name].min()
            max_val = data[col_name].max()
            mean_val = data[col_name].mean()
            median_val = data[col_name].median()
            std_val = data[col_name].std()
            data[col_name].fillna(0, inplace=True)
            print(col_name + str(median_val))
            # if imputate_type == "normalization":
            #     data[col_name] = (data[col_name].to_numpy() - min_val) / (max_val - min_val)
            # elif imputate_type == "standardlization":
            #     data[col_name] = (data[col_name].to_numpy() - mean_val) / std_val

        if "SUV_feimen" in col_name:
            mode_val = data[col_name].mode()[0]
            min_val = data[col_name].min()
            max_val = data[col_name].max()
            mean_val = data[col_name].mean()
            median_val = data[col_name].median()
            std_val = data[col_name].std()
            data[col_name].fillna(0, inplace=True)
            print(col_name + str(median_val))
            # if imputate_type == "normalization":
            #     data[col_name] = (data[col_name].to_numpy() - min_val) / (max_val - min_val)
            # elif imputate_type == "standardlization":
            #     data[col_name] = (data[col_name].to_numpy() - mean_val) / std_val



    datatype_dict = {
        "Image_id": "int32",
        "CT_exam_id2": "int32",
        "Stage_cT": "category",
        "Stage_cN": "category",
        "主肿物良恶性": "category",
        "N1淋巴结良恶性": "category",
        "N2淋巴结良恶性": "category",
        "主肿物位置": "category",
        "主肿物长径": "float32",
        "主肿物短径": "float32",
        "主肿物形状_毛刺": "int32",
        "主肿物形状_毛糙": "int32",
        "主肿物形状_分叶": "int32",
        "主肿物磨玻璃或实性": "category",
        "是否侵犯血管": "int32",
        "是否侵犯胸膜": "int32",
        "是否侵犯支气管": "int32",
        "肺门淋巴结长径": "int32" if binarize_size else "float32",
        "肺门淋巴结短径": "int32" if binarize_size else "float32",
        "纵隔淋巴结长径": "int32" if binarize_size else "float32",
        "纵隔淋巴结短径": "int32" if binarize_size else "float32",
        "Age": "float32",
        "Smoking_history": "int32",
        "Drinking_history": "int32",
        "Family_tumor_history": "int32",
        "Gender": "int32",
        "Pretreatment_CEA1": "float32",
        "Pretreatment_CA1991": "float32",
        "Pretreatment_CA1251": "float32",
        "Pretreatment_NSE1": "float32",
        "Pretreatment_CYFRA2111": "float32",
        "Pretreatment_SCC1": "float32",
        "Pretreatment_PETCT_tumor_SUVmax": "float32",
        "SUV_zongge": "int32" if binarize_SUVmax else "float32",
        "SUV_feimen": "int32" if binarize_SUVmax else "float32",
        "高血压": "int32",
        "糖尿病": "int32",
        "肺结核": "int32",
        "心血管疾病": "int32",
        "脑血管疾病": "int32",
                    
    }

    deepfeature = [str(x) for x in range(2,10)]
    for col_name in data.columns:
        if col_name in deepfeature:
            continue
        data = data.astype({col_name: datatype_dict[col_name]})


    return data


def feature_normalization(data: pd.DataFrame, normalization_type="normalization"):
    data = copy.deepcopy(data)
    deepfeature = [str(x) for x in range(2, 130)]
    for col_name in data.columns:
        if col_name in deepfeature:
            continue
        if pd.api.types.is_float_dtype(data[col_name].dtypes):
            mode_val = data[col_name].mode()[0]
            min_val = data[col_name].min()
            max_val = data[col_name].max()
            mean_val = data[col_name].mean()
            std_val = data[col_name].std()
            data[col_name].fillna(mean_val, inplace=True)
            if normalization_type == "normalization":
                data[col_name] = (data[col_name].to_numpy() - min_val) / (max_val - min_val)
            elif normalization_type == "standardlization":
                data[col_name] = (data[col_name].to_numpy() - mean_val) / std_val

    return data


def feature_selection(data:pd.DataFrame, labels, significance=0.05, show_results=False):
    data_fs = copy.deepcopy(data)
    for col_name in data.columns:
        if pd.api.types.is_categorical_dtype(data[col_name].dtypes) or pd.api.types.is_integer_dtype(data[col_name].dtypes):
            contingency_tab = pd.crosstab(data[col_name], labels)
            chi2, pvalue, _, _ = chi2_contingency(contingency_tab.values)
            if show_results:
                print(col_name)
                print(contingency_tab)
                print('P-value:{:.3f}'.format(pvalue))
                print('\n')

            if pvalue >= significance: # and '肺结核' not in col_name
                data_fs.drop(columns=col_name, inplace=True)
                # print("The feature {} is drop because its chi2 p-value is {} >= {}".format(col_name, pvalue, significance))


        elif pd.api.types.is_float_dtype(data[col_name].dtypes):
            group_neg = data[col_name].values[labels==0]
            group_pos = data[col_name].values[labels==1]
            statistic, pvalue = mannwhitneyu(group_neg, group_pos)
            if show_results:
                print(col_name)
                print('ALL mean:{:.2f}, std:[{:.2f}, {:.2f}]'.format(np.mean(data[col_name].values), np.mean(data[col_name].values)-np.std(data[col_name].values), np.mean(data[col_name].values)+np.std(data[col_name].values)))
                print('POS mean:{:.2f}, std:[{:.2f}, {:.2f}]'.format(np.mean(group_pos), np.mean(group_pos)-np.std(group_pos), np.mean(group_pos)+np.std(group_pos)))
                print('NEG mean:{:.2f}, std:[{:.2f}, {:.2f}]'.format(np.mean(group_neg), np.mean(group_neg)-np.std(group_neg), np.mean(group_neg)+np.std(group_neg)))
                print('P-value:{:.3f}'.format(pvalue))
                print('\n')
            if pvalue >= significance:
                data_fs.drop(columns=col_name, inplace=True)
                # print("The feature {} is drop because its mannwhitne p-value is {} >= {}".format(col_name, pvalue, significance))

    return data_fs


def one_hot_encoding(data:pd.DataFrame, drop_first=False):
    # dummy_data = pd.get_dummies(data=data, columns=["主肿物形状_毛刺",
    #                                                 "主肿物形状_分叶",
    #                                                 "Smoking_history",
    #                                                 "Drinking_history",
    #                                                 "Family_tumor_history",
    #                                                 "Gender"],
    #                             drop_first=drop_first)
    dummy_data = pd.get_dummies(data=data, drop_first=drop_first)

    return dummy_data
