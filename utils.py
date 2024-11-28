import subprocess
import fiftyone as fo
import fiftyone.brain as fob
import torch
import numpy as np
import threading
from PIL import Image
import clip
import io
import sys

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
    def __init__(self, port):
        self.tensorboard_process = None
        self.port = port

    def start(self):
        # 텐서보드 프로세스가 이미 실행 중인지 확인
        if self.tensorboard_process is None or self.tensorboard_process.poll() is not None:
            self.tensorboard_process = subprocess.Popen(
                ["tensorboard", "--logdir=runs/train", "--port={}".format(self.port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

    def emit_event(self, socketio, event_name, event_data):
        socketio.emit(event_name, event_data)
        print(f"Emitted '{event_name}' event")

    def stop(self):
        if self.tensorboard_process:
            self.tensorboard_process.terminate()
            self.tensorboard_process.wait()\
    
class FiftyoneManager:
    def __init__(self, dataset_dir, dataset_name, dataset_type, port):
        self.dataset = None
        self.fiftyone_thread = None
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.port = port

    def emit_event(self, socketio, event_name, event_data):
        socketio.emit(event_name, event_data)
        print(f"Emitted '{event_name}' event")

    # 데이터셋 로드 및 세션 생성
    def start(self):
        # 기존 데이터셋 삭제 (있는 경우)
        if fo.dataset_exists(self.dataset_name):
            print(f"Open existing dataset : '{self.dataset_name}'")
            dataset = fo.load_dataset(self.dataset_name)
            dataset_type = dataset.tags[0]

        else:
            if dataset_type == "51":
                dataset_type = "FiftyOneDataset"
                # 데이터셋 로드
                dataset = fo.Dataset.from_dir(
                    dataset_dir=self.dataset_dir,
                    dataset_type=fo.types.FiftyOneDataset,
                    name=self.dataset_name,
                )
                dataset.tags.append(dataset_type)

            elif dataset_type == "yolo":
                dataset_type = "YOLOv5Dataset"
                splits = ['train', 'val', 'test']
                dataset = fo.Dataset(self.dataset_name)

                for split in splits:
                    dataset.add_dir(
                        dataset_dir=self.dataset_dir,
                        dataset_type=fo.types.YOLOv5Dataset,
                        split=split,
                        tags=split,
                    )
                dataset.tags.append(dataset_type)
            
            elif dataset_type == "raw_image":
                dataset_type = "RawImageData"
                dataset = fo.Dataset.from_images_dir(self.dataset_dir)
                dataset.name = self.dataset_name
                dataset.tags.append(dataset_type)

            # 데이터셋을 영구적으로 저장
            dataset.persistent = True
            dataset.save()

            # CLIP 모델 로드
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)

            # 임베딩 생성 함수 : 이미지, 텍스트 구분
            def get_embeddings(sample):

                if "image" in sample.tags:
                    with torch.no_grad():
                        inputs = preprocess(Image.open(sample.filepath)).unsqueeze(0).to(device)
                        features = model.encode_image(inputs)

                # elif sample.tags[1] == "text":
                #     with torch.no_grad():
                #         inputs = clip.tokenize(sample.original_text, context_length=77, truncate=True).to(device)
                #         features = model.encode_text(inputs)

                return features.cpu().numpy().flatten()
            
            # 이미지 임베딩 추출
            embeddings = []
            cnt = 0
            for sample in dataset:
                sample.tags.append("image")
                embedding = get_embeddings(sample)
                embeddings.append(embedding)
                cnt += 1
                if cnt % 50 == 0:
                    print(f"{cnt}개 완료")
                
            embeddings = np.array(embeddings)

            # 임베딩을 데이터셋에 추가
            for sample, embedding in zip(dataset, embeddings):
                sample['clip_embeddings'] = embedding.tolist()
                sample.save()

            # 임베딩 시각화
            results = fob.compute_visualization(
                dataset,
                embeddings=embeddings,
                pathes_field="clip_embeddings",
                brain_key="clip_embeddings",
                # num_dims=3,
                plot_points=True,
                verbose=True,
            )

        print(f"Loaded dataset '{dataset.name}' with {len(dataset)} samples")

        # FiftyOne 세션 실행
        def run_fiftyone_session():
            session = fo.launch_app(dataset, port=self.port)
            session.wait()

        # 스레드 생성 및 실행
        fiftyone_thread = threading.Thread(target=run_fiftyone_session)
        fiftyone_thread.start()

        return fiftyone_thread, dataset, dataset_type
    
class LabelingManager:
    pass