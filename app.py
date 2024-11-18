import fiftyone as fo
import fiftyone.brain as fob
import clip
import torch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from flask import Flask, render_template_string, request, redirect, url_for, Response, send_file

import os
import sys
import argparse
from tqdm import tqdm
import threading
import io
import time
from trainer import train_yolo

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_dir", type=str, required=True, help="Importing dataset path")
parser.add_argument("--dataset_name", type=str, default="imported_dataset", help="Name the dataset you are importing")
parser.add_argument("--port", type=int, default=5151, help="Port to run the FiftyOne app on")
parser.add_argument("--dataset_type", type=str, default=None, help="dataset type (51, yolo)")

args = parser.parse_args()

# 데이터셋 로드 및 세션 생성은 애플리케이션 시작 시 한 번만 수행
dataset = None
fiftyone_thread = None

# 데이터셋 로드 및 세션 생성
def initialize_dataset_and_session():
    global dataset, fiftyone_thread

    # 기존 데이터셋 삭제 (있는 경우)
    if fo.dataset_exists(args.dataset_name):
        print(f"Open existing dataset : '{args.dataset_name}'")
        dataset = fo.load_dataset(args.dataset_name)
        dataset_type = dataset.tags[0]

    else:
        if args.dataset_type == "51":
            dataset_type = "FiftyOneDataset"
            # 데이터셋 로드
            dataset = fo.Dataset.from_dir(
                dataset_dir=args.dataset_dir,
                dataset_type=fo.types.FiftyOneDataset,
                name=args.dataset_name,
            )
            dataset.tags.append(dataset_type)

        elif args.dataset_type == "yolo":
            dataset_type = "YOLOv5Dataset"
            splits = ['train', 'val', 'test']
            dataset = fo.Dataset(args.dataset_name)

            for split in splits:
                dataset.add_dir(
                    dataset_dir=args.dataset_dir,
                    dataset_type=fo.types.YOLOv5Dataset,
                    split=split,
                    tags=split,
                )
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
            num_dims=3,
            plot_points=True,
            verbose=True,
        )

    print(f"Loaded dataset '{dataset.name}' with {len(dataset)} samples")

    # FiftyOne 세션 실행
    def run_fiftyone_session():
        session = fo.launch_app(dataset, port=args.port)
        session.wait()

    # 스레드 생성 및 실행
    fiftyone_thread = threading.Thread(target=run_fiftyone_session)
    fiftyone_thread.start()

    return dataset_type

# # 애플리케이션 시작 시 데이터셋과 세션 초기화
# dataset_type = initialize_dataset_and_session()

app = Flask(__name__)

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

## logger instance
capture_stream = CaptureOutput()
sys.stdout = capture_stream


@app.route('/get_views', methods=['GET'])
def get_views():
    # Fetch the list of saved views
    list_views = dataset.list_saved_views()
    return {'views': list_views}

@app.route('/')
def home():
    # 데이터셋 뷰 목록 조회
    list_views = dataset.list_saved_views()

    # HTML template with an iframe
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DA Framework Test App</title>
        <style>
            .button-container {{
                display: none; /* save button hidden : if you want to show, change display: flex */
                justify-content: flex-end;
                margin-top: 20px;
                margin-bottom: 20px;
            }}
            .dropdown-container {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 20px;
                margin-bottom: 20px;
            }}
            .dropdowns {{
                display: flex;
                align-items: center;
                gap: 10px;
                width: 100%;
            }}
            .export-button, .save-button {{
                font-size: 20px;
                padding: 10px 20px;
            }}
            .view-dropdown, .format-dropdown {{
                font-size: 20px;
                padding: 10px;
                flex-grow: 1;
            }}
            .go2trainer-button {{
                font-size: 24px;
                padding: 15px 30px;
                bottom: 20px;
                left: 20px;
                margin-top: 40px;
            }}
        </style>
        <script>
            function updateViews() {{
                fetch('/get_views')
                    .then(response => response.json())
                    .then(data => {{
                        const dropdown = document.getElementById('views');
                        dropdown.innerHTML = '';
                        data.views.forEach(view => {{
                            const option = document.createElement('option');
                            option.value = view;
                            option.textContent = view;
                            dropdown.appendChild(option);
                        }});
                    }});
            }}
        </script>
    </head>
    <body>
        <iframe src="http://localhost:{args.port}" width="90%" height="800px" frameborder="0"></iframe>
        <div class="button-container">
            <form action="/save" method="post">
                <button type="submit" class="save-button">Save Dataset</button>
            </form>
        </div>
        <div class="dropdown-container">
            <form action="/export" method="post" style="flex-grow: 1;">
                <div class="dropdowns">
                    <label for="views">뷰 선택:</label>
                    <select id="views" name="selected_view" class="view-dropdown">
                        {''.join(f'<option value="{view}">{view}</option>' for view in list_views)}
                    </select>
                    <label for="format">포맷 선택:</label>
                    <select id="format" name="selected_format" class="format-dropdown">
                        <option value="FiftyOneDataset">FiftyOneDataset</option>
                        <option value="YOLOv5Dataset">YOLOv5Dataset</option>
                    </select>
                    <button type="submit" class="export-button">Export Dataset</button>
                </div>
            </form>
        </div>
        <form action="/train_page" method="get">
            <button type="submit" class="go2trainer-button">Train Model</button>
        </form>
    </body>
    </html>
    """
    return render_template_string(html)

# @app.route('/save', methods=['POST'])
# def save_dataset():
#     # 데이터셋 변경사항 저장 및 내보내기
#     print()
#     print("Save Changes of Dataset...")
#     print()
#     # export_dir = args.dataset_dir
#     # dataset.export(
#     #     export_dir=export_dir,
#     #     dataset_type=fo.types.FiftyOneDataset,
#     # )
#     dataset.save()

#     return redirect(url_for('home'))

@app.route('/export', methods=['POST'])
def export_selected_view():
    # 선택된 뷰 가져오기
    selected_view = request.form.get('selected_view')
    selected_format = request.form.get('selected_format')
    print(f"Selected view: {selected_view}, Selected format: {selected_format}")  # 디버깅을 위한 출력

    if selected_view and selected_format:
        print(f"Exporting View: {selected_view}")
        view = dataset.load_saved_view(selected_view)
        view_export_dir = f"./datasets/exported_datasets/{dataset.name}_{selected_view}"
        label_field = "ground_truth"

        # 전체 샘플을 train, val로 스플릿
        splits = ['train', 'val']
        split_ratios = [0.8, 0.2]  # 예시 비율
        view.shuffle(seed=42)  # 랜덤 시드로 셔플

        # 기존의 train, val, test 태그 삭제 및 새로 추가
        num_samples = len(view)
        split_indices = np.cumsum([int(r * num_samples) for r in split_ratios])

        for idx, sample in enumerate(view):
            # 기존 태그 중 train, val, test 제거
            sample.tags = [tag for tag in sample.tags if tag not in splits]

            # 새로운 스플릿 태그 추가
            if idx < split_indices[0]:
                sample.tags.append('train')
            elif idx < split_indices[1]:
                sample.tags.append('val')
            sample.save()

        # 내보내기 포맷 선택
        if selected_format == "YOLOv5Dataset":
            dataset_type = fo.types.YOLOv5Dataset
        elif selected_format == "FiftyOneDataset":
            dataset_type = fo.types.FiftyOneDataset

        # 내보내기
        for split in splits:
            split_view = view.match_tags(split)
            split_view.export(
                export_dir=view_export_dir,
                dataset_type=dataset_type,
                label_field=label_field,
                split=split,
            )

        print(f"Exported to {view_export_dir}")

    return redirect(url_for('home'))

@app.route('/train_page', methods=['GET', 'POST'])
def train_page():
    # Exported datasets directory
    export_dir = './datasets/exported_datasets/'
    # Models directory
    models_dir = './models/'

    # 데이터셋과 모델의 경로를 저장
    datasets = [os.path.join(export_dir, d) for d in os.listdir(export_dir) if os.path.isdir(os.path.join(export_dir, d))]
    models = [os.path.join(models_dir, m) for m in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, m))]

    # HTML template for the train page
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Train Model</title>
        <style>
            .train-container {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-top: 50px;
                padding: 0 50px;
            }}
            .table-container {{
                display: flex;
                align-items: center;
                gap: 10px;
                width: 100%;
            }}
            .dataset-table, .model-table {{
                width: 90%;
                border-collapse: collapse;
                font-size: 25px;
            }}
            .dataset-table th, .dataset-table td, .model-table th, .model-table td {{
                border: 3px solid #ddd;
                padding: 20px;
                text-align: left;
                vertical-align: middle; /* 수직 중앙 정렬 */
            }}
            .dataset-table th, .model-table th {{
                background-color: #f2f2f2;
            }}
            .train-button {{
                font-size: 24px;
                padding: 15px 30px;
                align-self: flex-start;
            }}
            .download-button {{
                font-size: 24px;
                padding: 15px 30px;
                margin-top: 20px;
                background-color: #4CAF50; /* Green */
                color: white;
                border: none;
                cursor: pointer;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                position: absolute;
                right: 50px;
            }}
            input[type="radio"] {{
                transform: scale(2.0); /* 라디오 버튼 크기 조정 */
                margin-right: 10px;
            }}
            #log {{
                width: 100%;
                height: 300px;
                overflow-y: scroll;
                border: 1px solid #ddd;
                padding: 10px;
                font-family: monospace;
                background-color: #f9f9f9;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="train-container">
            <form action="/train" method="post" style="flex-grow: 1;">
                <div class="table-container">
                    <table class="dataset-table">
                        <thead>
                        <tr>
                            <th>Select</th>
                            <th>Dataset Name</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(f'<tr><td><input type="radio" name="selected_dataset" value="{dataset}"></td><td>{os.path.basename(dataset)}</td></tr>' for dataset in datasets)}
                        </tbody>
                    </table>
                    <table class="model-table">
                        <thead>
                            <tr>
                                <th>Select</th>
                                <th>Model Name</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(f'<tr><td><input type="radio" name="selected_model" value="{model}"></td><td>{os.path.basename(model)}</td></tr>' for model in models)}
                        </tbody>
                    </table>
                    <button type="submit" class="train-button">Start Training</button>
                </div>
            </form>
        </div>
        <div id="log"></div> <!-- 로그 출력 영역 추가 -->
        <a href="/download_model" class="download-button">Download Model</a> <!-- 다운로드 버튼 추가 -->
        <script>
            function fetchLogs() {{
                const eventSource = new EventSource('/stream_logs');
                const logElement = document.getElementById('log');
                eventSource.onmessage = function(event) {{
                    logElement.innerHTML += event.data + '<br>';
                    logElement.scrollTop = logElement.scrollHeight;
                }};
            }}
            window.onload = fetchLogs;
        </script>
    </body>
    </html>
    """

    return render_template_string(html)

@app.route('/train', methods=['POST'])
def train():

    selected_dataset = request.form.get('selected_dataset') + '/dataset.yaml'
    selected_model = request.form.get('selected_model')
    
    # 별도의 스레드에서 훈련 시작
    training_thread = threading.Thread(target=train_yolo, args=(selected_dataset, selected_model))
    training_thread.start()
    training_thread.join()

    return redirect(url_for('train_page'))

@app.route('/download_model')
def download_model():
    model_path = 'runs/train/weights/best.pt'
    return send_file(model_path, as_attachment=True)

@app.route('/stream_logs')
def stream_logs():
    def generate():
        while True:
            line = capture_stream.get_output()
            if line:
                yield f"data: {line}\n\n"
                capture_stream.clear_output() # 로그 전송 후 초기화
            time.sleep(1)  # Add a small delay to prevent high CPU usage

    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":

    dataset_type = initialize_dataset_and_session()
    app.run(port=5555, debug=False)
    fiftyone_thread.join()