import json
import time
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# 모델과 토크나이저 로드 (bge-m3 모델 사용)
model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to('cuda')

# ChromaDB에서 사용할 HuggingFace 임베딩 함수
embedding_function = HuggingFaceEmbeddings(model_name=model_name)

# 텍스트를 임베딩하는 함수
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

# 벡터 정규화 함수 (코사인 유사도를 위해)
def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# dataset.json 파일에서 데이터 불러오기
with open('test_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
start_time = time.time()
# 각 문서의 input만을 추출하여 벡터로 변환 및 정규화
documents = [Document(page_content=doc['context']) for doc in data]  # Document 객체로 변환
document_embeddings = embed_text([doc['context'] for doc in data], tokenizer, model)
document_embeddings_normalized = normalize(document_embeddings)

# ChromaDB 인스턴스 생성
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
# 검색 시간 출력.
print(f"임베딩 시간: {emb_time}초")
print(f"검색 시간: {result_time}초")

# 검색된 문서 출력
for result in results:
    print(f"문서: {result.page_content}")
