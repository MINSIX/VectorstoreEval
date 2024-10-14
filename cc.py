import json
import time
import faiss
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from kobert_tokenizer import KoBERTTokenizer
# 모델과 토크나이저 로드 (bge-m3 모델 사용)

# google-bert/bert-base-multilingual-cased
# snunlp/KR-SBERT-V40K-klueNLI-augSTS
# "kobert": AutoModel.from_pretrained('skt/kobert-base-v1'),
# BAAI/bge-m3
model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer = KoBERTTokenizer.from_pretrained(model_name, sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})
model = AutoModel.from_pretrained(model_name).to('cuda')

# 데이터를 임베딩하는 함수 정의
def embed_text(texts, tokenizer, model, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: val.to('cuda') for key, val in inputs.items()}  # 데이터를 GPU로 전송
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# 벡터를 정규화하는 함수 정의 (코사인 유사도를 위해 필요)
def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# dataset.json 파일에서 데이터 불러오기
with open('test_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 검색 시간 측정
start_time = time.time()

# 각 문서의 input만을 추출하여 벡터로 변환
documents = [doc['context'] for doc in data]
document_embeddings = embed_text(documents, tokenizer, model)

# 코사인 유사도를 위한 벡터 정규화
document_embeddings_normalized = normalize(document_embeddings)

# FAISS 내적 기반 인덱스 생성
dimension = document_embeddings_normalized.shape[1]
index = faiss.IndexFlatIP(dimension)  # 내적(Dot Product) 기반 검색
index.add(np.array(document_embeddings_normalized))

end_time = time.time()
emb_time = end_time - start_time

# 검색 쿼리
query = "교통사고 관련 이슈를 알려주세요"
query_embedding = embed_text([query], tokenizer, model)
query_embedding_normalized = normalize(query_embedding)

# 검색 시간 측정
start_time = time.time()
k = 5  # 상위 5개 문서 검색
distances, indices = index.search(np.array(query_embedding_normalized), k)
end_time = time.time()

# 검색 시간 출력
search_time = end_time - start_time
print(f"생성 시간: {emb_time}초")
print(f"검색 시간: {search_time}초")

# 검색된 문서 및 유사도 출력 (내적 값 자체가 코사인 유사도임)
print("\n상위 검색 문서들:")
for i, idx in enumerate(indices[0]):
    print(f"문서 {i+1}: {documents[idx]} (유사도: {distances[0][i]})")
