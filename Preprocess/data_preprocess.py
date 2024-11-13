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

# 定義財務和保險資料的檔案路徑
finance_path = [f"/home/davidteng/aicup/jage_rag/data/reference/finance/{i}.pdf" for i in range(0, 1035)]
insurance_path = [f"/home/davidteng/aicup/jage_rag/data/reference/insurance/{i}.pdf" for i in range(1, 644)]

# 初始化財務和保險的原始資料列表
finance_raw_data = []
insurance_raw_data = []

# 從 PDF 檔案中提取財務資料
for path in tqdm(finance_path):
	elements = partition_pdf(filename=path)
	finance_raw_data.append("\n".join([str(el) for el in elements]))

# 從 PDF 檔案中提取保險資料
for path in tqdm(insurance_path):
	elements = partition_pdf(filename=path)
	insurance_raw_data.append("\n".join([str(el) for el in elements]))

# 讀取已存在的 JSON 檔案，提取財務和保險資料
with open("data.json", "r", encoding="utf-8") as f:
	data = json.load(f)

# 獲取 finance 和 insurance 資料
finance_raw_data = data["finance"]
insurance_raw_data = data["insurance"]

# 確認資料數量是否正確
print(len(finance_path), len(finance_raw_data))
print(len(insurance_path), len(insurance_raw_data))

# 設定文本分割參數
chunk_size = 200
chunk_overlap = 100
finance_chunks = []
insurance_chunks = []

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
	chunk_size=chunk_size,  # 每個段落的最大字數
	chunk_overlap=chunk_overlap,  # 每個段落之間的重疊字數
	separators=["\n\n", "\n", "，", "。", "；"]  # 優先使用段落、空行、空格等作為分割依據
)

def convert_year_3(match):
	"""
	將民國年份轉換為西元年份。
	"""
	roc_year = int(match.group(3))
	ad_year = roc_year + 1911
	return f"西元{ad_year}年"

def convert_year_5(match):
	"""
	將民國年份轉換為西元年份。
	"""
	roc_year = int(match.group(3))
	ad_year = roc_year + 1911
	return f"西元{ad_year}年"

def convert_chinses_num(match):
	"""
	將中文數字轉換為阿拉伯數字。
	"""
	chinese_num = match.group(0)
	try:
		return str(cn2an.cn2an(chinese_num, "normal"))
	except Exception:
		return chinese_num

# 分割和轉換財務資料
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

# 分割和轉換保險資料
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

# 將處理後的資料寫入 JSON 檔案
with open(f"..//chunks/finance_chunks_{chunk_size}_{chunk_overlap}.json", "w", encoding="utf-8") as f:
	json.dump(finance_chunks, f, ensure_ascii=False, indent=4)

with open(f"..//chunks/insurance_chunks_{chunk_size}_{chunk_overlap}.json", "w", encoding="utf-8") as f:
	json.dump(insurance_chunks, f, ensure_ascii=False, indent=4)