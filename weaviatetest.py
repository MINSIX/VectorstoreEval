import json
import time
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import weaviate

from kobert_tokenizer import KoBERTTokenizer
# google-bert/bert-base-multilingual-cased
# snunlp/KR-SBERT-V40K-klueNLI-augSTS
# "kobert": AutoModel.from_pretrained('skt/kobert-base-v1'),
# BAAI/bge-m3
# model_name = "skt/kobert-base-v1"
# # tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer = KoBERTTokenizer.from_pretrained(model_name, sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})
model_name = "skt/kobert-base-v1"
tokenizer = KoBERTTokenizer.from_pretrained(model_name, sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})

# tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Weaviate 클라이언트 연결
client = weaviate.Client("http://localhost:8080")

# 클래스 정의
class_name = "Document"
if not client.schema.exists(class_name):
    class_obj = {
        "class": class_name,
        "vectorizer": "none",
        "properties": [
            {"name": "context", "dataType": ["string"]},
        ]
    }
    client.schema.create_class(class_obj)

# 모델과 토크나이저 로드 (bge-m3 모델 사용)

# 텍스트를 임베딩하는 함수
def embed_text(texts, tokenizer, model, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# 벡터 정규화 함수 (코사인 유사도 사용을 위해)
def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# 데이터 로드
with open('test_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 각 문서의 텍스트 추출 및 임베딩
start_time = time.time()
documents = [doc['context'] for doc in data]
document_embeddings = embed_text(documents, tokenizer, model)

# 벡터 정규화
document_embeddings_normalized = normalize(document_embeddings)

# Weaviate에 데이터 삽입
for i, doc in enumerate(documents):
    client.data_object.create(
        data_object={"context": doc},
        class_name=class_name,
        vector=document_embeddings_normalized[i].tolist()
    )

end_time = time.time()
emb_time = end_time - start_time
print(f"임베딩 시간: {emb_time}초")

# 검색 쿼리 생성 및 임베딩
query = "교통사고 관련 이슈를 알려주세요"
query_embedding = embed_text([query], tokenizer, model)

# 쿼리 벡터 정규화
query_embedding_normalized = normalize(query_embedding)

# 검색 시간 측정
start_time = time.time()

# 검색 쿼리 실행 (distance 값을 1.0으로 설정하고 limit 값을 20으로 늘림)
result = client.query.get(class_name, ["context"]).with_near_vector({
    "vector": query_embedding_normalized[0].tolist(),
    "distance": 5.0  # 검색 범위를 최대화
}).with_limit(100).do()  # 더 많은 문서를 검색하여 중복 없는 5개 선택

end_time = time.time()

# 검색 시간 출력
print(f"검색 시간: {end_time - start_time}초")

# 검색된 문서 출력, 중복 제거 후 최대 5개의 고유한 문서만 출력
retrieved_docs = set()  # 중복된 문서를 추적하기 위한 집합
doc_count = 0

if 'data' in result and 'Get' in result['data'] and class_name in result['data']['Get']:
    docs = result['data']['Get'][class_name]
    
    if docs:
        print("\n상위 검색 문서들:")
        for res in docs:
            if res['context'] not in retrieved_docs:
                print(f"문서: {res['context']}")
                retrieved_docs.add(res['context'])  # 중복된 문서를 추적
                doc_count += 1
            if doc_count == 5:  # 5개의 문서가 출력되면 중지
                break
    else:
        print("검색된 문서가 없습니다.")
else:
    print("검색 결과가 비어 있습니다.")
