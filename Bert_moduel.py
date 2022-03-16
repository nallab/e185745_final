from os import environ, replace
import matplotlib.pyplot as plt
import pandas as pd
import tabula
import glob
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AlbertTokenizer
from sklearn.model_selection import KFold
import pytorch_lightning as pl

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

#ファイルの読み込み
def load_data(txt_name):
  dfs = tabula.read_pdf(txt_name, lattice=True, pages='all')
  #all_dataに全ページのデータ格納
  all_data = pd.DataFrame()
  pd.set_option('display.max_rows', 300)
  #カラムの"Unnamed"をNaNに変更
  for index,df in enumerate(dfs):
      for i,col in enumerate(df.columns):
          if "Unnamed:" in col:
              df = df.rename(columns={col: " "})
      dfs[index] = df

  for i in range(len(dfs)):
      #全ページのデータを結合させる
      head_data = list(dfs[i].columns)
      df = pd.DataFrame(head_data).T
      dfs[i] = dfs[i].set_axis(df.columns,axis=1)
      df = df.append(dfs[i])
      all_data = all_data.append(df)
  all_data = all_data.set_axis(range(len(all_data)),axis=0).fillna(" ")

  for k in reversed(range(len(all_data))):
      if all_data.iloc[k,1] == " ":
          all_data.iloc[k-1,2] = all_data.iloc[k-1,2] + " " + all_data.iloc[k,2]
          all_data.iloc[k-1,3] = all_data.iloc[k-1,3] + " " + all_data.iloc[k,3]
          all_data = all_data.drop(all_data.index[k],axis=0)

  for index in range(len(all_data)):
      all_data.iloc[index,3] = str(all_data.iloc[index,3]).split("\r")[0]
      all_data.iloc[index,3] = str(all_data.iloc[index,3]).split(" ")[0]
      all_data.iloc[index,3] = all_data.iloc[index,3].replace("-","").replace(",","")
      
      all_data.iloc[index,2] = all_data.iloc[index,2].replace("-","").replace(",","").replace("\r"," ")
  counselor = list(all_data[all_data[1] == "I"][2])
  code = list(all_data[all_data[1] == "I"][3])
  return counselor,code

def concat():
    file_list = sorted(glob.glob("MitiData/NoHeader/*"))
    counselors = []
    codes = []
    for name in file_list:
        counselors.extend(load_data(name)[0])
        codes.extend(load_data(name)[1])
        #counselors = counselors[:int(len(counselors)*0.7)]
        #codes = codes[:int(len(codes)*0.7)]
    return counselors, codes

def load_nonlabel():
    file_list = sorted(glob.glob("MitiData/HighLowQualityCounseling/transcripts/*"))
    C = []
    T = []
    for name in file_list:
        with open(name,"r") as f:
            data = f.readlines()
        for i,t in enumerate(data):
            txt = t.replace("\t","").replace("\n","").split(":")
            if txt[0] == "C":
                C.append(txt[1])
            elif txt[0] == "T":
                T.append(txt[1])
            else:
                continue
    return T

def change_code(code_list):
    category_list = ["GI","PERSUADE","PERSUADE WITH",'Q',"SR","CR","AF","SEEK","EMPHASIZE","CONFRONT","NC"]
    #category_list = ["GI","PERSUADE",'Q',"SR","CR","AF","SEEK","EMPHASIZE","CONFRONT","NC"]
    for i,miti in enumerate(code_list):
        code_list[i] = category_list.index(miti)
    return code_list


def make_tensor(corpus,labeldata):
  loader = []
  for text,label in zip(corpus,labeldata):
    encoding = tokenizer(
    text,
    max_length=512, 
    padding='max_length'
    )
    encoding['labels'] = label # ラベルを追加
    encoding = { k: torch.tensor(v, device=0) for k, v in encoding.items() }
    loader.append(encoding)
  return loader

