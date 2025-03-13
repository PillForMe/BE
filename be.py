import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import requests
from flask_cors import CORS
import concurrent.futures
from umap import UMAP
from sklearn.cluster import KMeans

# 문장 임베딩 함수 정의
def embed_sentence(sentence, model):
    return model.embed_query(sentence)

def parallel_embedding(sentences, model, num_threads=32):
    embeddings = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_sentence = {executor.submit(embed_sentence, sentence, model): sentence for sentence in sentences}
        for future in concurrent.futures.as_completed(future_to_sentence):
            sentence = future_to_sentence[future]
            try:
                embedding = future.result()
                embeddings.append(embedding)
            except Exception as exc:
                print(f"Sentence {sentence} generated an exception: {exc}")
    return embeddings

# Pinecone 객체 생성 및 API 키 설정
pc = Pinecone(api_key=api_key)
# Pinecone index 이름 설정 및 생성
index_name = "adv-db-1"
index = pc.Index(name=index_name)

# open_ai 임베딩
embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

app = Flask(__name__)
CORS(app)  # 모든 도메인에서의 요청을 허용

bad_reviews = pd.read_excel('./부정_건강고민_일반특성정보.xlsx')
good_reviews = pd.read_excel('./긍정_건강고민_일반특성정보.xlsx')
item_info = pd.read_excel('./item_info.xlsx')

# ga_result = [['동아제약_셀파렉스 에센셜포우먼', '오리진_옵티멈 오메가-3 맥스 1200'], ['세노비스_마그네슘', '센트룸_우먼(국내)'], ['동아제약_셀파렉스 에센셜포우먼', '파마젠_알티지 오메가3'], ['커클랜드_데일리 멀티 비타민&미네랄(국내)', '21세기센트리_마그네슘 250mg'], ['센트룸_우먼(국내)', '레이크에비뉴뉴트리션_마그네슘 비스글리시네이트 킬레이트 200mg'], ['커클랜드_데일리 멀티 비타민&미네랄(국내)', '레이크에비뉴뉴트리션_마그네슘 비스글리시네이트 킬레이트 200mg'], ['커클랜드_데일리 멀티 비타민&미네랄(국내)', '블루보넷뉴트리션_킬레이트 마그네슘'], ['블루보넷뉴트리션_킬레이트 칼슘 마그네슘', '커클랜드_데일리 멀티 비타민&미네랄(국내)'], ['센트룸_우먼(국내)', '캘리포니아골드뉴트리션_마그네슘 킬레이트'], ['나우푸드_데일리 비츠 (캡슐)', '21세기센트리_마그네슘 250mg'], ['블루보넷뉴트리션_버퍼드 킬레이트 마그네슘', '센트룸_우먼(국내)'], ['세노비스_마그네슘', '나우푸드_데일리 비츠 (캡슐)'], ['나우스포츠_ZMA', '나우푸드_데일리 비츠 (캡슐)'], ['커클랜드_데일리 멀티 비타민&미네랄(국내)', '캘리포니아골드뉴트리션_마그네슘 킬레이트']]

# selected_good_elements = []
# selected_bad_elements = []
# result = []
# all_health = []
user = None

@app.route('/ga_result', methods=['POST'])
def ga_result():
    global  user

    good_health_reviews = []
    bad_health_reviews = []

    incoming_data = request.json  # 프론트엔드로부터 받은 데이터
    # print('Received from frontend:', incoming_data)

    user = incoming_data['user']

    url = 'http://127.0.0.1:6000/ga'
    response = requests.post(url, json=incoming_data)
    try:
        data = response.json()
        ga_output = data['ga_output']
    except requests.exceptions.JSONDecodeError:
        return jsonify({"error": "Invalid JSON response from /ga endpoint"}), 500

    all_health = []

    for combo in ga_output:
        combo_info = []
        for item in combo:
            health_info = item_info.loc[item_info['브랜드명_제품명'] == item, '건강 고민 정보'].values
            combo_info.append((item, list(health_info)))
            all_health.extend(health_info[0].split(","))
        # result.append(combo_info)

    all_health = list(set(all_health))

    for health in all_health:
        good_health_info = good_reviews.loc[good_reviews['건강고민'] == health, '일반특성정보'].values
        if (len(list(good_health_info)) > 0):
            good_health_reviews.extend(good_health_info[0].split('\n'))
        
        bad_health_info = bad_reviews.loc[bad_reviews['건강고민'] == health, '일반특성정보'].values
        if (len(list(bad_health_info)) > 0):
            bad_health_reviews.extend(bad_health_info[0].split('\n'))
        
    good_health_reivews = list(set(good_health_reviews))
    bad_health_reivews = list(set(bad_health_reviews))

    # 병렬 처리로 좋은 리뷰와 나쁜 리뷰 임베딩
    good_embeddings = parallel_embedding(good_health_reivews, embeddings_model)
    bad_embeddings = parallel_embedding(bad_health_reivews, embeddings_model)

    # print(len(good_embeddings))

    umap = UMAP(n_components=2, random_state=42)
    umap_good_embeddings = umap.fit_transform(np.stack(good_embeddings))
    umap_bad_embeddings = umap.fit_transform(np.stack(bad_embeddings))

    # KMeans 군집화
    num_clusters = 24  # 원하는 군집의 수
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)

    kmeans.fit(umap_good_embeddings)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    selected_good_elements = []
    selected_bad_elements = []

    # 각 군집에서 가장 대표성을 띠는 요소를 선택
    for i in range(num_clusters):
        cluster_elements = np.where(labels == i)[0]
        if len(cluster_elements) > 0:
            # 군집의 중심점과 각 요소 간의 거리 계산
            distances = np.linalg.norm(np.array(umap_good_embeddings)[cluster_elements] - centroids[i], axis=1)
            # 가장 가까운 요소 선택
            closest_element_index = cluster_elements[np.argmin(distances)]
            selected_good_elements.append(good_health_reviews[closest_element_index])
    
    chunk_size = 4
    selected_good_elements = [selected_good_elements[i:i + chunk_size] for i in range(0, len(selected_good_elements), chunk_size)]

    kmeans.fit(umap_bad_embeddings)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # 각 군집에서 가장 대표성을 띠는 요소를 선택
    for i in range(num_clusters):
        cluster_elements = np.where(labels == i)[0]
        if len(cluster_elements) > 0:
            # 군집의 중심점과 각 요소 간의 거리 계산
            distances = np.linalg.norm(np.array(umap_bad_embeddings)[cluster_elements] - centroids[i], axis=1)
            # 가장 가까운 요소 선택
            closest_element_index = cluster_elements[np.argmin(distances)]
            selected_bad_elements.append(bad_health_reviews[closest_element_index])

    selected_bad_elements = [selected_bad_elements[i:i + chunk_size] for i in range(0, len(selected_bad_elements), chunk_size)]

    return_result = {
        "goods" : selected_good_elements,
        "bads" : selected_bad_elements,
        "all_health" : all_health,
        "ga_output" : ga_output
    }

    return jsonify(return_result), 200


# @app.route('/send_infos', methods = ['GET'])
# def reviews_embed():

#     return_result = {
#         "selected_good_elements" : selected_good_elements,
#         "selected_bad_elements" : selected_bad_elements,
#         "all_health" : all_health
#     }

#     return jsonify(return_result), 200

@app.route('/glrec_result', methods=['POST'])
def glrec_result():
    # global ga_output

    incoming_data = request.json  # 백엔드로부터 받은 데이터
    # print('Received from BE:', incoming_data)

    user_data = user
    efficacy_data= incoming_data["efficacy"]
    ga_output = incoming_data["gaOutput"]

    # print(user_data)
    # print(efficacy_data)
    # print(ga_output)

    send_data = {
        "user" : user_data,
        "efficacy" : efficacy_data,
        "ga_output" : ga_output
    }
    url = 'http://127.0.0.1:7000/glrec'
    response = requests.post(url, json=send_data)
    try:
        data = response.json()
        glrec_result = data['glrec_result']
    except requests.exceptions.JSONDecodeError:
        return jsonify({"error": "Invalid JSON response from /ga endpoint"}), 500
    
    return_result = {"glrec_result" : glrec_result}

    return jsonify(return_result), 200
    

@app.route('/final_result', methods=['POST'])
def final_result():
    incoming_data = request.json

    glrec_result = incoming_data["glRecResult"]
    selected_goods_data = incoming_data["selectedGoods"]
    selected_bads_data = incoming_data["selectedBads"]
    ga_output = incoming_data["gaOutput"]

    review_weight = 0.1

    # print("ga_output : ", ga_output)
    print("glrec_result : ", glrec_result)
    # print("selected_goods_data : ", selected_goods_data)
    # print("selected_bads_data : ", selected_bads_data)

    items = []
    for comb in ga_output:
        for item in comb:
            items.append(item)
    items = list(set(items))
    query_metadata = []
    for item in items:
        query_metadata.append({'브랜드명_제품명' : item})

    i=0
    for goods in selected_goods_data:
        query_question = goods
        query_embedding = embeddings_model.embed_query(query_question)
        query_embedding_np = np.array(query_embedding).astype(np.float32)
        query_embedding_list = query_embedding_np.tolist()
    
        # Pinecone 쿼리 실행
        response = index.query(
            vector=query_embedding_list,   # 쿼리 벡터
            top_k=10,                      # 상위 3개의 결과를 반환
            include_metadata=True,        # 메타데이터 포함
            filter={
                '$or': query_metadata,
                '리뷰 종류': 'good' 
            }
        )

        # print("good_response : ", response)

        matches = []
        for match in response['matches']:
            try:
                matches.append(match['metadata']['브랜드명_제품명'])
            except:
                pass
        matches = list(set(matches))
        for match in matches:
            glrec_result[match] +=review_weight*(6-i)
        i+=1

    i=0
    for bads in selected_bads_data:
        query_question = bads
        query_embedding = embeddings_model.embed_query(query_question)
        query_embedding_np = np.array(query_embedding).astype(np.float32)
        query_embedding_list = query_embedding_np.tolist()

        # Pinecone 쿼리 실행
        response = index.query(
            vector=query_embedding_list,   # 쿼리 벡터
            top_k=10,                      # 상위 3개의 결과를 반환
            include_metadata=True,        # 메타데이터 포함
            filter={
                '$or': query_metadata,
                '리뷰 종류': 'bad' 
            }
        )

        # print("bad_response : ", response)

        matches = []
        for match in response['matches']:
            try:
                matches.append(match['metadata']['브랜드명_제품명'])
            except:
                pass
        matches = list(set(matches))
        for match in matches:
            glrec_result[match] -=review_weight*(6-i)
        i+=1

    # print("ga_output : ", ga_output)


    final_result = []
    for comb in ga_output:
        score = 0
        for item in comb:
            if item in glrec_result:  # item이 glrec_result에 있는지 확인
                score += glrec_result[str(item)]
            else:
                print(f"Warning: {item} not found in glrec_result")  # 디버깅을 위한 경고 메시지 출력
        final_result.append((comb, score))

    # final_result를 score에 대해 내림차순으로 정렬
    final_result = sorted(final_result, key=lambda x: x[1], reverse=True)

    print("final_result : ", [list(item) for item in final_result])

    result = {
        "final_result" : final_result
    }

    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)