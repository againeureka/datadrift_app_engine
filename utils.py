import fiftyone as fo
import fiftyone.brain as fob
from pymilvus import MilvusClient
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

import clip
import torch
import numpy as np
from PIL import Image

import threading
import io
import sys
import os
from tqdm import tqdm
import subprocess
import time

class MilvusManager:
    def __init__(self):
        self.client = None

    def connect(self, db_file_path):
        print(f"Connecting to {db_file_path}")
        if not os.path.exists("db"):
            print(f"Creating New Database : {db_file_path}")
            os.makedirs("db")
            self.client = MilvusClient(os.path.join("db", db_file_path))
        else:
            print(f"Connecting to Existing Database : db/{db_file_path}")
            self.client = MilvusClient(os.path.join("db", db_file_path))
        
    def create_collection(self, collection_name):
        if self.client.has_collection(collection_name):
            print(f"Dropping existing collection {collection_name}")
            self.client.drop_collection(collection_name)

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="sample_id", datatype=DataType.VARCHAR, is_primary=True, max_length=24)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=512)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type='AUTOINDEX',
            metric_type="COSINE",
        )
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        print(f"Collection {collection_name} created successfully {self.client.get_load_state(collection_name)}")
        print(self.client.describe_collection(collection_name=collection_name))

    def has_collection(self, collection_name):
        return self.client.has_collection(collection_name)
    
    def drop_collection(self, collection_name):
        self.client.drop_collection(collection_name)
    
    def insert(self, collection_name, data):
        start_time = time.time()
        print(f"Inserting datas to {collection_name}")
        self.client.insert(
            collection_name=collection_name,
            data=data,
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print(f"{self.client.get_collection_stats(collection_name)} // Entities inserted successfully.")
        print(f"Time taken: {formatted_time}")
        
    def getdata_by_id(self, collection_name, ids):
        query_results = self.client.get(
            collection_name=collection_name,
            ids=ids,
            output_fields=["sample_id", "embedding"],
        )
        return query_results
        

## stdout logger class
class CaptureOutput(io.StringIO):
    def __init__(self, max_length=100):
        super().__init__()
        self.output = []
        self.max_length = max_length  # 로그를 유지할 최대 줄 수 설정
        self.auto_clear_threshold = 100  # 자동으로 클리어할 줄 수 설정

    def write(self, txt):
        super().write(txt)
        sys.__stdout__.write(txt)  # 터미널에도 출력
        lines = txt.splitlines()
        for line in lines:
            if line:
                self.output.append(line)
                # 로그 줄 수가 최대 길이를 초과하면 가장 오래된 로그부터 제거
                while len(self.output) > self.max_length:
                    self.output.pop(0)
                # 로그 줄 수가 자동 클리어 임계값을 초과하면 로그를 초기화
                if len(self.output) > self.auto_clear_threshold:
                    self.clear_output()

    def get_output(self):
        return '\n'.join(self.output)

    def clear_output(self):
        self.output = []

class TensorboardManager:
    def __init__(self, port=6006):
        self.tensorboard_process = None
        self.port = port
        self.process = None
        self.current_logdir = None

    def start(self, logdir=None):
        if logdir:
            self.current_logdir = logdir
        elif not self.current_logdir:
            self.current_logdir = "runs"  # 기본값

        if self.process:
            self.stop()

        command = f"tensorboard --logdir={self.current_logdir} --port={self.port}"
        self.process = subprocess.Popen(command.split())

    def emit_event(self, socketio, event_name, event_data):
        socketio.emit(event_name, event_data)
        print(f"Emitted '{event_name}' event")

    def stop(self):
        if self.tensorboard_process:
            self.tensorboard_process.terminate()
            self.tensorboard_process.wait()\
    
class FiftyoneManager:
    def __init__(self, port):
        self.fiftyone_thread = None
        self.session = None
        self.port = port

    def emit_event(self, socketio, event_name, event_data):
        socketio.emit(event_name, event_data)
        print(f"Emitted '{event_name}' event")

    # 데이터셋 로드 및 세션 생성
    def start(self):

        print(f"Opening FiftyOne Session on port {self.port}")
        
        # FiftyOne 세션 실행
        def run_fiftyone_session():

            self.session = fo.launch_app(port=self.port)
            self.session.wait()

        # 스레드 생성 및 실행
        self.fiftyone_thread = threading.Thread(target=run_fiftyone_session)
        self.fiftyone_thread.start()

        return self.fiftyone_thread

    def set_dataset(self, dataset, vis_results):
        
        if self.session:
            results = vis_results
            self.session.dataset = dataset

    # 임베딩 생성 함수 : 이미지, 텍스트 구분
    def get_embeddings(self, dataset, device, model, preprocess):
        data = []
        for sample in tqdm(dataset, desc=f"{dataset.name} 임베딩 계산 중"):
            if "image" in sample.tags:
                with torch.no_grad():
                    inputs = preprocess(Image.open(sample.filepath)).unsqueeze(0).to(device)
                    embedding = model.encode_image(inputs).cpu().numpy().flatten()
                    tmp_dict = {
                        "sample_id": sample.id,
                        "embedding": embedding.tolist()
                    }
                    data.append(tmp_dict)

        # elif sample.tags[1] == "text":
        #     with torch.no_grad():
        #         inputs = clip.tokenize(sample.original_text, context_length=77, truncate=True).to(device)
        #         features = model.encode_text(inputs)

        return data
    
    def collect_image_embeddings_by_sample_id(self, data, db_client=None):
        embeddings_by_sample_id = {}
        
        if isinstance(data, list):
            for d in tqdm(data, desc="시각화용 데이터 처리 중"):
                embeddings_by_sample_id[d['sample_id']] = np.array(d['embedding'])

        elif isinstance(data, fo.core.dataset.Dataset):
            for sample in tqdm(data, desc="시각화용 데이터 처리 중"):
                results = db_client.getdata_by_id(data.name, [sample.id])
                embeddings_by_sample_id[sample.id] = np.array(results[0]['embedding'])

        else:
            raise ValueError("Invalid data type")
        
        print(f"Collected {len(embeddings_by_sample_id)} embeddings")
        return embeddings_by_sample_id

class InputDataLoader:
    def __init__(self, data_path, data_type, data_name=None):
        self.data_path = data_path
        self.data = data_name
        self.data_type = data_type
        self.dataset = None

    def get_img_data(self):
        
        if self.data_type == "FiftyOneDataset":
            # 데이터셋 로드
            self.dataset = fo.Dataset.from_dir(
                dataset_dir=self.data_path,
                dataset_type=fo.types.FiftyOneDataset,
                name=self.data,
            )
            self.dataset.tags.append(self.data_type)

        elif self.data_type == "YOLOv5Dataset":
            self.dataset = fo.Dataset(self.data)
            
            # 두 가지 가능한 구조 확인
            structure_1 = os.path.exists(os.path.join(self.data_path, 'images'))  # images/split/
            structure_2 = os.path.exists(os.path.join(self.data_path, 'train'))   # split/images/

            if structure_1:
                print("Detected structure: images/split/")
                # images 디렉토리 내의 split 확인
                available_splits = [d for d in os.listdir(os.path.join(self.data_path, 'images')) 
                                  if os.path.isdir(os.path.join(self.data_path, 'images', d))]
                print(f"Available splits: {available_splits}")
                
                for split in available_splits:
                    try:
                        print(f"Adding {split} dataset to {self.dataset.name}")
                        self.dataset.add_dir(
                            dataset_dir=self.data_path,
                            dataset_type=fo.types.YOLOv5Dataset,
                            split=split,
                            tags=split,
                        )
                    except Exception as e:
                        print(f"Error adding {split} dataset: {e}")

            elif structure_2:
                print("Detected structure: split/images/")
                # 루트 디렉토리 내의 split 확인
                available_splits = [d for d in os.listdir(self.data_path) 
                                  if os.path.isdir(os.path.join(self.data_path, d))]
                print(f"Available splits: {available_splits}")
                
                for split in available_splits:
                    try:
                        print(f"Adding {split} dataset to {self.dataset.name}")
                        self.dataset.add_dir(
                            dataset_dir=self.data_path,
                            dataset_type=fo.types.YOLOv5Dataset,
                            split=split,
                            tags=split,
                        )
                    except Exception as e:
                        print(f"Error adding {split} dataset: {e}")
            else:
                raise Exception("Could not detect valid YOLOv5 dataset structure. Expected either 'images/split/' or 'split/images/'")
                
            self.dataset.tags.append(self.data_type)
        
        elif self.data_type == "RawImageData":
            self.dataset = fo.Dataset.from_images_dir(self.data_path)
            self.dataset.name = self.data
            self.dataset.tags.append(self.data_type)

        return self.dataset
    
    # 개별 데이터 태깅, 메타데이터 추가
    def add_tags(self, source):

        for sample in self.dataset:
            sample.tags.append("image")
            sample['source'] = source
            sample.save()


class LabelingManager:
    pass