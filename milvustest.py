import json
import time
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility

from kobert_tokenizer import KoBERTTokenizer
# google-bert/bert-base-multilingual-cased
# snunlp/KR-SBERT-V40K-klueNLI-augSTS
# "kobert": AutoModel.from_pretrained('skt/kobert-base-v1'),
# BAAI/bge-m3
# 
# # tokenizer = AutoTokenizer.from_pretrained(model_name)
# # model_name = "skt/kobert-base-v1"
# tokenizer = KoBERTTokenizer.from_pretrained(model_name, sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})

# 모델과 토크나이저 로드 (bge-m3 모델 사용)
model_name = "skt/kobert-base-v1"
tokenizer = KoBERTTokenizer.from_pretrained(model_name, sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})

model = AutoModel.from_pretrained(model_name)

# 텍스트를 임베딩하는 함수 정의
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

# Milvus에 연결
connections.connect("default", host="localhost", port="19530")

# 필드 스키마 정의 (임베딩 차원 수정)
id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)
embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)  # bge-m3 모델의 임베딩 크기

# 컬렉션 스키마 생성
collection_name = "document_collection"
schema = CollectionSchema(fields=[id_field, embedding_field], description="Document collection schema")

# 기존 컬렉션이 있으면 삭제 후 새로 생성
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

collection = Collection(name=collection_name, schema=schema)

# dataset.json 파일에서 데이터 불러오기
with open('test_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 1. 데이터 삽입 시간 측정
start_insertion_time = time.time()
# 각 문서의 텍스트를 추출하고 임베딩 변환
documents = [doc['context'] for doc in data]
start_time = time.time()
document_embeddings = embed_text(documents, tokenizer, model)
print(f"Embedding shape: {document_embeddings.shape}")

# 벡터 정규화 (코사인 유사도 사용을 위해)
document_embeddings_normalized = normalize(document_embeddings)

ids = list(range(len(document_embeddings)))
entities = [ids, document_embeddings_normalized.tolist()]  # IDs와 임베딩 데이터를 준비
collection.insert(entities)

collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist":500}})


# 3. 컬렉션을 메모리에 로드
collection.load()
end_index_time = time.time()
index_creation_time = end_index_time - start_insertion_time
print(f"인덱스 생성 시간: {index_creation_time}초")
# 4. 검색 성능 테스트 (검색 시간 측정)
query = "교통사고 관련 이슈를 알려주세요"
query_embedding = embed_text([query], tokenizer, model)

# 쿼리 벡터 정규화
query_embedding_normalized = normalize(query_embedding)

# 임베딩 차원이 맞는지 확인
print(f"Query embedding shape: {query_embedding_normalized.shape}")

# 검색 시간 측정
start_search_time = time.time()

# nprobe 값을 10에서 20으로 늘려 검색 범위 확대
search_params = {"metric_type": "IP", "params": {"nprobe": 500}}
results = collection.search(query_embedding_normalized, anns_field="embedding", param=search_params, limit=5, output_fields=["id"])

end_search_time = time.time()
search_time = end_search_time - start_search_time
print(f"컬렉션에 삽입된 데이터 수: {collection.num_entities}")

print(f"검색 시간: {search_time}초")

# 5. 검색된 문서 및 유사도 출력
if results:
    print("\n상위 검색 문서들:")
    for result in results:
        for res in result:
            idx = res.id
            print(f"문서 {idx}: {documents[idx]} (유사도: {res.distance})")
else:
    print("검색 결과가 비어 있습니다.")

# Milvus 연결 종료
connections.disconnect("default")
