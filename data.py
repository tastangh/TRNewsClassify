import pandas as pd
from sklearn.model_selection import train_test_split

def get_ttc4900():
    
    ttc4900 = pd.read_parquet("hf://datasets/savasy/ttc4900/ttc4900/train-00000-of-00001.parquet")
        
    # Categoryler 7 tane. her birisi 0,1,...7 olarak yazılmıs, bunlar güncelleniyor.

    category_mapping = {
        0: "siyaset",
        1: "dunya",
        2: "ekonomi",
        3: "kültür",
        4: "saglik",
        5: "spor",
        6: "teknoloji"
    }

    ttc4900["category"] = ttc4900["category"].replace(category_mapping)
    print(ttc4900.head())
    train_data_ttc4900, test_data_ttc4900 = train_test_split(ttc4900, test_size=0.2, random_state=42)
    
    return [train_data_ttc4900, test_data_ttc4900]
    
def get_trsav1():
    
    trsav1 = pd.read_csv("hf://datasets/maydogan/TRSAv1/TRSAv1.csv")
    trsav1_5k = trsav1.sample(n=5000, random_state=42)
    train_data_trsav1, test_data_trsav1 = train_test_split(trsav1_5k, test_size=0.2, random_state=42)
    
    return [train_data_trsav1, test_data_trsav1]
    