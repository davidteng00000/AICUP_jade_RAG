"""
NOTE:
This file is originally a .ipynb file.
According to the requirements, we rewrite it into .py file.
Hence, there may be some inconvenience and informality.
To have more detiail, please check "data_preprocess.ipynb" in the same folder.
"""


import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from tqdm import tqdm
import cn2an


finance_path = [f"/home/davidteng/aicup/jage_rag/data/reference/finance/{i}.pdf" for i in range(0, 1035)]
insurance_path = [f"/home/davidteng/aicup/jage_rag/data/reference/insurance/{i}.pdf" for i in range(1, 644)]


print(finance_path[0:10])


finance_raw_data = []
insurance_raw_data = []


for path in tqdm(finance_path):
    elements = partition_pdf(filename=path)
    finance_raw_data.append("\n".join([str(el) for el in elements]))
    

for path in tqdm(insurance_path):
    elements = partition_pdf(filename=path)
    insurance_raw_data.append("\n".join([str(el) for el in elements]))
    

# with open("data.json", "w", encoding="utf-8") as f:
#     json.dump({
#         "finance": finance_raw_data,
#         "insurance": insurance_raw_data
#     }, f, ensure_ascii=False, indent=4)


with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 獲取 finance 和 insurance 資料
finance_raw_data = data["finance"]
insurance_raw_data = data["insurance"]


print(len(finance_path), len(finance_raw_data))
print(len(insurance_path), len(insurance_raw_data))


chunk_size = 200
chunk_overlap = 100


finance_chunks = []
insurance_chunks = []


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,  # 每個段落的最大字數
    chunk_overlap=chunk_overlap,  # 每個段落之間的重疊字數
    separators=["\n\n", "\n", "，", "。", "；"]  # 優先使用段落、空行、空格等作為分割依據
)
# chunks = text_splitter.split_text(text)


def convert_year_3(match):
    # 提取民國年中的數字部分並轉換為整數
    roc_year = int(match.group(3))
    # 西元年 = 民國年 + 1911
    ad_year = roc_year + 1911
    return f"西元{ad_year}年"

def convert_year_5(match):
    # 提取民國年中的數字部分並轉換為整數
    roc_year = int(match.group(3))
    # 西元年 = 民國年 + 1911
    ad_year = roc_year + 1911
    return f"西元{ad_year}年"

def convert_chinses_num(match):
    chinese_num = match.group(0)
    try:
        # 使用 cn2an 進行轉換並返回字符串結果
        return str(cn2an.cn2an(chinese_num, "normal"))
    except Exception:
        # 若轉換失敗，返回原始字符串
        return chinese_num


id = 0
for i, text in enumerate(finance_raw_data, 0):
    chunks = text_splitter.split_text(text)
    for chunk in chunks:
        chunk = chunk.replace(' ', '')
        chunk = chunk.replace('○', '零')
        chunk = re.sub(r"[○零一二三四五六七八九十百千萬億]{2,}", convert_chinses_num, chunk)
        chunk = re.sub(r"(民國)*?([\\n\s])*?((?<!\d)\d{2,3}(?=\D))([\\n\s])*?年", convert_year_5, chunk)
        chunk = re.sub(r"(民國)([\\n\s])*?(\d{2,3})([\\n\s])*?(年)*?", convert_year_3, chunk)
        finance_chunks.append(
            {
                "text": chunk,
                "source": i,
                "id": id
            }
        )
        id += 1


id = 0
for i, text in enumerate(insurance_raw_data, 1):
    chunks = text_splitter.split_text(text)
    for chunk in chunks:
        chunk = chunk.replace(' ', '')
        chunk = re.sub(r"[零一二三四五六七八九十百千萬億]{2,}", convert_chinses_num, chunk)
        chunk = re.sub(r"(民國)*?([\\n\s])*?((?<!\d)\d{2,3}(?=\D))([\\n\s])*?年", convert_year_5, chunk)
        chunk = re.sub(r"(民國)([\\n\s])*?(\d{2,3})([\\n\s])*?(年)*?", convert_year_3, chunk)
        insurance_chunks.append(
            {
                "text": chunk,
                "source": i,
                "id": id
            }
        )
        id += 1


with open(f"..//chunks/finance_chunks_{chunk_size}_{chunk_overlap}.json", "w", encoding="utf-8") as f:
    json.dump(finance_chunks, f, ensure_ascii=False, indent=4)
with open(f"..//chunks/insurance_chunks_{chunk_size}_{chunk_overlap}.json", "w", encoding="utf-8") as f:
    json.dump(insurance_chunks, f, ensure_ascii=False, indent=4)