"""
NOTE:
This file is originally a .ipynb file.
According to the requirements, we rewrite it into .py file.
Hence, there may be some inconvenience and informality.
To have more detiail, please check "retrieval.ipynb" in the same folder.
""" 



"""
Env Settings
"""
from sentence_transformers import SentenceTransformer, util
import json
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import jieba
# from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer

IS_CREATING_DATABASE = False



"""
Parameters
"""
embedding_topk = 10
bm25_topn = 10
docs_top = 1

# output_file = f"../data/test_result/result.json"
name = 'final_run2_301-600'
r_path = f'../data/test_result/result_{name}.json'
wa_path = f'../data/test_result/wa_{name}.json'
log_path = f'../logs/{name}.json'

tmp_path = './tmp.json' # 錯誤時存檔處

out_path = f'../outputs/pred_retrieve_final_run2_301-600.json'  # 輸出繳交檔案格式的路徑
q_path = '../data/dataset/preliminary/questions_preliminary.json' # 問題路徑

notes = ''

prompt = """你是一個RAG 檢索篩選機器人，你會根據query 以及chunks 列表輸出一個最能正確回答query 的 chunk ID。\n每一個chunks 前後都有<|start_chunk_X|> 和 <|end_chunk_X|> 標籤，其中X代表chunk ID。\n輸入：query、chunks 列表\n輸出：一個最能正確回答query 的 chunk ID，無論如何一定要輸出一個，不能輸出'沒有找到相關的chunk來回答這個query'等等語句。\n你的回答必須符合下列格式與規範：\n1. 禁止greeting \n2. 只輸出一個數字，禁止輸出任何其他符號或是文字\n範例輸出：15"""



"""
Load Model
"""
model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')



"""
Create Database (skip)
"""
if IS_CREATING_DATABASE:
	G_faq = dict()
	G_finance = dict()
	G_insurance = dict()



"""
Loading Docs & Docs into Embeddings (skip)
- 長文切分方式
- text2embedding的方式
"""
with open('../chunks/finance_chunks_400_200.json', 'r', encoding='utf-8') as f:
	data = json.load(f)
with open('./fc_test.json', 'r', encoding='utf-8') as f:
	fc_data = json.load(f)
data += fc_data
for entry in tqdm(data):
	text = entry.get('text')
	text = str(text)
	entry_id = entry.get('source')
	chunk_id = entry.get('id')
	if text is not None and entry_id is not None:
		embedding = model.encode(text, convert_to_tensor=True)
		# doc_inputs = context_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
		# embedding = context_encoder(**doc_inputs).pooler_output
		if entry_id not in G_finance:
			G_finance[entry_id] = []
		G_finance[entry_id].append( {'text':text,'embedding':embedding, 'id': chunk_id} )
	else:
		print(f"Missing 'text' or 'id' in entry: {entry}")


with open('../chunks/insurance_chunks_400_200.json', 'r', encoding='utf-8') as f:
	data = json.load(f)
for entry in tqdm(data):
	text = entry.get('text')
	entry_id = entry.get('source')
	chunk_id = entry.get('id')
	if text is not None and entry_id is not None:
		embedding = model.encode(text, convert_to_tensor=True)
		# doc_inputs = context_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
		# embedding = context_encoder(**doc_inputs).pooler_output
		if entry_id not in G_insurance:
			G_insurance[entry_id] = []
		G_insurance[entry_id].append( {'text':text,'embedding':embedding, 'id': chunk_id} )
	else:
		print(f"Missing 'text' or 'id' in entry: {entry}")


def merge_question_answers(data):
	question = data.get('question')
	answers = data.get('answers')
	return [ f"問題：{question} 答案：{answer}" for  answer in answers] if question and answers else None


file_path = "../data/reference/faq/pid_map_content.json"
with open(file_path, 'r', encoding='utf-8') as f:
	data = json.load(f)
chunk_id = 0
for entry_id, lst in tqdm(data.items()):
	for entry in lst:
		texts = merge_question_answers(entry)
		for text in texts:
			if text is not None and entry_id is not None:
				embedding = model.encode(text, convert_to_tensor=True)
				# doc_inputs = context_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
				# embedding = context_encoder(**doc_inputs).pooler_output
				if int(entry_id) not in G_faq:
					G_faq[int(entry_id)] = []
				G_faq[int(entry_id)].append({'text':text,'embedding':embedding, 'id': chunk_id})
				chunk_id += 1
			else:
				print(f"Missing 'text' or 'id' in entry: {entry}")


for key,value in G_insurance.items():
	print(key,value)
	break


import pickle

# 將字典存入檔案
with open("v2_400_200_table.pkl", "wb") as file:
	pickle.dump((G_faq, G_finance, G_insurance), file)



"""
load
"""
import pickle

# 從檔案中讀取字典
with open("v5_400_200_table.pkl", "rb") as file:
	G_faq, G_finance, G_insurance = pickle.load(file)


chunk_dict = {
	'faq':{
		
	},
	'finance':{
		
	},
	'insurance':{
		
	}
}
for key,value in G_faq.items():
	for chunk in value:
		chunk_dict['faq'][chunk.get('id')] = {
			'text': chunk.get('text'),
			'source': key
		}
for key,value in G_finance.items():
	for chunk in value:
		chunk_dict['finance'][chunk.get('id')] = {
			'text': chunk.get('text'),
			'source': key
		}
for key,value in G_insurance.items():
	for chunk in value:
		chunk_dict['insurance'][chunk.get('id')] = {
			'text': chunk.get('text'),
			'source': key
		}
print(chunk_dict['insurance'][21])



"""
Retrive Function
"""
from collections import Counter
def retrieve_documents(query, source_list, G, k=1, threshold=0.5):
	query_embedding = model.encode(query, convert_to_tensor=True)
	# query_inputs = question_tokenizer(query, return_tensors="pt")
	# query_embedding = question_encoder(**query_inputs).pooler_output
	# print(query, source_list)
	scores = []
	for entry_id in source_list:
		if entry_id not in G:
			continue
		for data in G.get(entry_id):
			score = util.pytorch_cos_sim(query_embedding, data.get('embedding'))[0].item()
			scores.append((entry_id, data.get('id'), score))
	# print(scores)
	
	sorted_scores = sorted(scores, key=lambda x: x[2], reverse=True)
	# print(sorted_scores)
	top_scores = sorted_scores[:k]
	return top_scores


def retrieve_documents_bm25(query, source_list, G, n = 1):
	# Tokenize the query and documents
	tokenized_query = list(jieba.cut_for_search(query))  # Simple tokenization, modify as needed
	corpus = []
	tokenized_corpus = []
	documents = []
	
	for entry_id in source_list:
		if entry_id not in G:
			continue
		for data in G.get(entry_id):
			corpus.append(data.get('text'))
			tokenized_corpus.append(list(jieba.cut_for_search(data.get('text'))))  # Tokenize each document
			documents.append(data.get('text'))
	
	# Initialize BM25
	bm25 = BM25Okapi(tokenized_corpus)
	
	ans_list = bm25.get_top_n(tokenized_query, list(documents), n)
	ans_id_list = set()
	
	for entry_id in source_list:
		if entry_id not in G:
			continue
		for data in G.get(entry_id):
			for ans in ans_list:
				if data.get('text') == ans:
					ans_id_list.add((entry_id, data.get('id')))

	return list(ans_id_list)


from openai import OpenAI
import os
api_key = 'sk-'
os.environ['OPENAI_API_KEY'] = api_key
def llm_chat(messages, model, temperature=0.5, max_tokens=10, top_p=1, frequency_penalty=0, presence_penalty=0, stop=[]):
	try:
		client = OpenAI(
			api_key=os.environ.get("OPENAI_API_KEY"),
		)
		response = client.chat.completions.create(
			model=model,
			messages=messages,
			temperature=temperature,
			max_tokens=max_tokens,
			top_p=top_p,
			frequency_penalty=frequency_penalty,
			presence_penalty=presence_penalty,
		)
		return response.choices[0].message.content
	except Exception as e:
		return f"error: {str(e)}"

def llm_select(query, options_list, prompt):
	query = f'query: {query}\n\nchunk list: {[i for i, source, chunk in options_list]}\n' + "\n".join([f"<|start_chunk_{i}|> {chunk} <|end_chunk_{i}|>" for i, source, chunk in (options_list)]) + f'你的輸出必須是{[i for i, source, chunk in options_list]}的其中一個數字'
	messages=[
		{
		"role": "system",
		"content": [
			{
			"type": "text",
			"text": prompt
			}
		]
		},
		{
		"role": "user",
		"content": [
			{
				"type": "text",
				"text": query
			}
		]
		}
	]
	chunk_id = llm_chat(messages, 'gpt-4o')
	return chunk_id

def find_overlap(str1, str2):
	max_overlap = 0
	for i in range(1, min(len(str1), len(str2)) + 1):
		if str1[-i:] == str2[:i]:
			max_overlap = i
	return max_overlap

def merge_overlap(my_set):
	lst = sorted(my_set, key=lambda x: x[0])
	ret_set = set()
	i = 0
	while i < len(lst) - 1:
		if lst[i][1] == lst[i + 1][1] and lst[i][0] + 1 == lst[i + 1][0]:
			str1 = lst[i][2]
			str2 = lst[i + 1][2]
			overlap_length = find_overlap(str1, str2)
			
			if overlap_length > 0:
				merged_text = str1 + str2[overlap_length:]
				lst[i] = (lst[i][0], lst[i][1], merged_text)
				
				del lst[i + 1]
			else:
				i += 1
		else:
			i += 1

	ret_set = {(item[0], item[1], item[2]) for item in lst}
	return ret_set

empty_set = set()
empty_set.add((3772, 392, "第27條匯款相關費用及其負擔對象\n本契約相關款項之往來，若因匯款而產生相關費用時，除下列各款約定所生之匯款相關費用均由本公司負擔外，匯款銀行及中間行所收取之相關費用，由匯款人負擔之，收款銀行所收取之收款手續費，由收款人負擔：\n第10頁，共16頁\n南山人壽威美鑽美元利率變動型終身壽險（定期給付型）_SYUL\n一、因可歸責於本公司之錯誤原因，致本公司依第30條第二項約定為退還或給\n付所生之相關匯款費用。\n二、因可歸責於本公司之錯誤原因，要保人或受益人依第30條第二項約定為補\n繳或返還所生之相關匯款費用。\n三、因本公司提供之匯款帳戶錯誤而使要保人或受益人匯款無法完成時所生之相\n關匯款費用。"))
empty_set.add((3773, 392, "第10頁，共16頁\n南山人壽威美鑽美元利率變動型終身壽險（定期給付型）_SYUL\n一、因可歸責於本公司之錯誤原因，致本公司依第30條第二項約定為退還或給\n付所生之相關匯款費用。\n二、因可歸責於本公司之錯誤原因，要保人或受益人依第30條第二項約定為補\n繳或返還所生之相關匯款費用。\n三、因本公司提供之匯款帳戶錯誤而使要保人或受益人匯款無法完成時所生之相\n關匯款費用。\n要保人或受益人若選擇以本公司指定銀行之外匯存款戶交付相關款項且匯款銀行及收款銀行為同一銀行時，或以本公司指定銀行之外匯存款戶受領相關款項時，其所有匯款相關費用均由本公司負擔，不適用前項約定。本公司指定銀行之相關訊息可至本公司網站（網址：http://www.nanshanlife.com.tw）查詢。\n第28條保險單借款及契約效力的停止"))
empty_set.add((3774, 392, "關匯款費用。\n要保人或受益人若選擇以本公司指定銀行之外匯存款戶交付相關款項且匯款銀行及收款銀行為同一銀行時，或以本公司指定銀行之外匯存款戶受領相關款項時，其所有匯款相關費用均由本公司負擔，不適用前項約定。本公司指定銀行之相關訊息可至本公司網站（網址：http://www.nanshanlife.com.tw）查詢。\n第28條保險單借款及契約效力的停止\n於本契約「保障期間」內，要保人得向本公司申請保險單借款，其可借金額上限為借款當日保單價值準備金之一定百分比，其比率請詳附表四，未償還之借款本息，超過其保單價值準備金時，本契約效力即行停止。但本公司應於效力停止日之30日前以書面通知要保人。本公司未依前項規定為通知時，於本公司以書面通知要保人返還借款本息之日起30日內要保人未返還者，保險契約之效力自該30日之次日起停止。\n第29條不分紅保單"))
result = merge_overlap(empty_set)
print("最終合併結果集合:", result)

def get_options(category, retrieved_embeddings, retrieved_bm25):
	option_list = set()
	e_set = list()
	bm25_set = list()
	
	for chunk in retrieved_embeddings:
		option_list.add((chunk[1], chunk_dict[category][chunk[1]]['source'], chunk_dict[category][chunk[1]]['text']))
		e_set.append((chunk[1], chunk_dict[category][chunk[1]]['source'], chunk_dict[category][chunk[1]]['text']))
	for chunk in retrieved_bm25:
		option_list.add((chunk[1], chunk_dict[category][chunk[1]]['source'], chunk_dict[category][chunk[1]]['text']))
		bm25_set.append((chunk[1], chunk_dict[category][chunk[1]]['source'], chunk_dict[category][chunk[1]]['text']))
		
	option_list = merge_overlap(option_list)
	return option_list, e_set, bm25_set

retrieve_documents_bm25('被保險人於本契約有效期間內身故，本公司是否會依本契約約定給付保險金？', [2], G_insurance, n = 1)


def run(q_path, embedding_topk, bm25_topn, docs_top, prompt, run_list, tmp_path):
	answer_dict = {"answers": []}
	with open(q_path, 'rb') as f:
		qs_ref = json.load(f)
	
	for q_dict in tqdm(qs_ref['questions']):
		try:
			if not q_dict['qid'] in run_list:
				continue
			if q_dict['category'] == 'finance':
				retrieved_embeddings = retrieve_documents(q_dict[ 'query'], q_dict['source'], G_finance, k = embedding_topk)
				retrieved_bm25 = retrieve_documents_bm25(q_dict[ 'query'], q_dict['source'], G_finance, n = bm25_topn)
				# print(retrieved_embeddings, retrieved_bm25)
				
				for s in q_dict['source']:
					retrieved_embeddings += retrieve_documents(q_dict[ 'query'], [s], G_finance, k = docs_top)
					retrieved_bm25 += retrieve_documents_bm25(q_dict[ 'query'], [s], G_finance, n = docs_top)
				# print(retrieved_embeddings, retrieved_bm25)
				option_list, e_set, bm25_set = get_options('finance', retrieved_embeddings, retrieved_bm25)
				
				selected_chunk = llm_select(q_dict['query'], option_list, prompt)
				retrieved = chunk_dict['finance'][int(selected_chunk)]['source']
				
				answer_dict['answers'].append({"qid": q_dict['qid'], "query": q_dict['query'], "retrieve": retrieved, "selected_chunk": (selected_chunk, chunk_dict['finance'][int(selected_chunk)]['text']), "option_list": list(option_list), "bm25_chunks": bm25_set, "embeddings_chunks": e_set})
				
			elif q_dict['category'] == 'insurance':
				retrieved_embeddings = retrieve_documents(q_dict[ 'query'], q_dict['source'], G_insurance, k = embedding_topk)
				retrieved_bm25 = retrieve_documents_bm25(q_dict[ 'query'], q_dict['source'], G_insurance, n = bm25_topn)
				# print(retrieved_embeddings, retrieved_bm25)
				for s in q_dict['source']:
					# print(s)
					retrieved_embeddings += retrieve_documents(q_dict[ 'query'], [s], G_insurance, k = docs_top)
					retrieved_bm25 += retrieve_documents_bm25(q_dict[ 'query'], [s], G_insurance, n = docs_top)
				
				option_list, e_set, bm25_set = get_options('insurance', retrieved_embeddings, retrieved_bm25)
				# print(retrieved_embeddings, retrieved_bm25)
				selected_chunk = llm_select(q_dict['query'], option_list, prompt)
				retrieved = chunk_dict['insurance'][int(selected_chunk)]['source']
				
				answer_dict['answers'].append({"qid": q_dict['qid'], "query": q_dict['query'], "retrieve": retrieved, "selected_chunk": (selected_chunk, chunk_dict['insurance'][int(selected_chunk)]['text']), "option_list": list(option_list), "bm25_chunks": bm25_set, "embeddings_chunks": e_set})        # print(retrieved)
			elif q_dict['category'] == 'faq':
				retrieved_embeddings = retrieve_documents(q_dict[ 'query'], q_dict['source'], G_faq, k = 7)
				retrieved_bm25 = retrieve_documents_bm25(q_dict[ 'query'], q_dict['source'], G_faq, n = 7)
				
				# for s in q_dict['source']:
				#     retrieved_embeddings += retrieve_documents(q_dict[ 'query'], [s], G_faq, k = docs_top)
				#     retrieved_bm25 += retrieve_documents_bm25(q_dict[ 'query'], [s], G_faq, n = docs_top)
				
				option_list, e_set, bm25_set = get_options('faq', retrieved_embeddings, retrieved_bm25)   

				selected_chunk = llm_select(q_dict['query'], option_list, prompt)
				retrieved = chunk_dict['faq'][int(selected_chunk)]['source']
				
				answer_dict['answers'].append({"qid": q_dict['qid'], "query": q_dict['query'], "retrieve": retrieved, "selected_chunk": (selected_chunk, chunk_dict['faq'][int(selected_chunk)]['text']), "option_list": list(option_list), "bm25_chunks": bm25_set, "embeddings_chunks": e_set})
			else:
				raise ValueError("Something went wrong")
		except Exception as e:
			print(f'error: {e}')
			try:
				ans_list = []
				ans_dict = {}
				result = answer_dict
				for sample in result['answers']:
					ans_list.append({
						"qid": sample['qid'],
						"retrieve": sample['retrieve']
					})
				ans_dict['answers'] = ans_list
				with open(tmp_path, 'w') as f:
					json.dump(ans_dict, f, ensure_ascii=False, indent=4)
				print(f'stored tmp file to {tmp_path}')
			except Exception as e:
				print(f'error storing tmp file: {e}')
			
	return answer_dict

import json

def calculate_accuracy(ground_truth_filename, pred_filename, wa_path):

	wrong_answer_list = []
	with open(ground_truth_filename, 'r') as f1, open(pred_filename, 'r') as f2:
		ground_truth_data = json.load(f1)
		pred_data = json.load(f2)

	correct_count = 0
	total_count = 0

	for gt in ground_truth_data['ground_truths']:
		for pred in pred_data['answers']:
			if gt['qid'] == pred['qid']:
				if gt['retrieve'] == pred['retrieve']:
					correct_count += 1
				else:
					wrong_answer_list.append({
						'qid': gt['qid'],
						'correct_answer': gt['retrieve'],
						'wrong_answer': pred['retrieve']
					})
				total_count += 1
	with open(wa_path, 'w', encoding='utf8') as f:
		json.dump(wrong_answer_list, f, ensure_ascii=False, indent=4)
	accuracy = correct_count / total_count
	return accuracy

def output_std_ans(r_path, out_path):
	ans_dict = {}
	ans_list = []
	with open(r_path, 'r') as f1:
		result = json.load(f1)
	for sample in result['answers']:
		ans_list.append({
			"qid": sample['qid'],
			"retrieve": sample['retrieve']
		})
	ans_dict['answers'] = ans_list
	with open(out_path, 'w') as f:
		json.dump(ans_dict, f, ensure_ascii=False, indent=4)



"""
Run
"""
run_list = [i for i in range(301, 601)]
# print(run_list)
# run_list = [64, 70, 82]
answer_dict = run(q_path, embedding_topk, bm25_topn, docs_top, prompt, run_list, tmp_path)

with open(r_path, 'w', encoding='utf8') as f:
	json.dump(answer_dict, f, ensure_ascii=False, indent=4)
output_std_ans(r_path, out_path)


"""
calculate acc
"""
# accuracy = calculate_accuracy('../data/dataset/preliminary/ground_truths_example.json', r_path, wa_path)
# logs(r_path, wa_path, log_path, embedding_topk,bm25_topn,docs_top,notes,prompt, accuracy)
# print(f'準確率：{accuracy}')