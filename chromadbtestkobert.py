import json
import time
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from kobert_tokenizer import KoBERTTokenizer

# KoBERT 모델과 토크나이저 로드
model_name = "skt/kobert-base-v1"
tokenizer = KoBERTTokenizer.from_pretrained(model_name, sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})
model = AutoModel.from_pretrained(model_name)  # CPU 모드에서 실행

# 텍스트를 임베딩하는 함수
def embed_text(texts, tokenizer, model, batch_size=8):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # KoBERT 모델의 출력 확인
        batch_embeddings = outputs.pooler_output.cpu().numpy() if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# 벡터 정규화 함수 (코사인 유사도를 위해)
def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# 데이터 로드
with open('test_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

start_time = time.time()
# 각 문서의 input을 벡터로 변환
documents = [Document(page_content=doc['context']) for doc in data]
document_embeddings = embed_text([doc['context'] for doc in data], tokenizer, model)
document_embeddings_normalized = normalize(document_embeddings)

# Hugging Face 임베딩 함수 사용
embedding_function = HuggingFaceEmbeddings(model_name=model_name)

# ChromaDB 인스턴스 생성 및 문서 추가
vector_store = Chroma.from_documents(documents, embedding_function)

end_time = time.time()

# 검색 쿼리
query = "교통사고 관련 이슈를 알려주세요"
query_embedding = embed_text([query], tokenizer, model)
query_embedding_normalized = normalize(query_embedding)

emb_time = end_time - start_time
# 검색 시간 측정
start_time = time.time()

# ChromaDB에서 유사도 검색
results = vector_store.similarity_search(query, k=5)

end_time = time.time()
result_time = end_time - start_time

# 검색 시간 출력
print(f"임베딩 시간: {emb_time}초")
print(f"검색 시간: {result_time}초")

# 검색된 문서 출력
for result in results:
    print(f"문서: {result.page_content}")
