import fiftyone as fo
import fiftyone.brain as fob
import clip
import torch

import numpy as np

from flask import Flask, render_template, request, redirect, url_for, Response, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS

import os
import sys
import argparse
from tqdm import tqdm
import threading
import io
import time
import subprocess
import atexit

from trainer import train_yolo
from utils import TensorboardManager, FiftyoneManager, CaptureOutput
parser = argparse.ArgumentParser()

parser.add_argument("--dataset_dir", type=str, required=True, help="Importing dataset path")
parser.add_argument("--dataset_name", type=str, default="imported_dataset", help="Name the dataset you are importing")
parser.add_argument("--port", type=int, default=5151, help="Port to run the FiftyOne app on")
parser.add_argument("--dataset_type", type=str, default=None, help="dataset type (51, yolo)")

args = parser.parse_args()

# 데이터셋 로드 및 세션 생성은 애플리케이션 시작 시 한 번만 수행
tsb_runner = TensorboardManager(port=6006)
atexit.register(tsb_runner.stop)

ffto_runner = FiftyoneManager(args.dataset_dir, args.dataset_name, args.dataset_type, args.port)
fiftyone_thread, dataset, dataset_type = ffto_runner.start()

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)

ffto_runner.emit_event(socketio, 'fiftyone_ready', {'status': 'ready'})

## logger instance
capture_stream = CaptureOutput()
sys.stdout = capture_stream

@app.route('/')
def index():
    return redirect(url_for('init_data'))

@app.route('/get_views', methods=['GET'])
def get_views():
    # Fetch the list of saved views
    list_views = dataset.list_saved_views()
    return {'views': list_views}

@app.route('/init_data')
def init_data():
    return render_template('init_data.html')

@app.route('/dataclinic')
def home():
    # 데이터셋 뷰 목록 조회
    list_views = dataset.list_saved_views()
    
    return render_template('home.html', list_views=list_views, port=args.port)

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

    return render_template('train_page.html', datasets=datasets, models=models)

@app.route('/train', methods=['POST'])
def train():

    selected_dataset = request.form.get('selected_dataset') + '/dataset.yaml'
    selected_model = request.form.get('selected_model')

    # 별도의 스레드에서 훈련 시작
    training_thread = threading.Thread(target=train_yolo, args=(selected_dataset, selected_model))
    training_thread.start()

    tensorboard_thread = threading.Thread(target=tsb_runner.start)
    tensorboard_thread.start()
    tsb_runner.emit_event(socketio, 'tensorboard_ready', {'status': 'ready'})

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

@socketio.on('check_fiftyone_ready')
def handle_check_fiftyone_ready():
    # FiftyOne 세션이 준비되었는지 확인하는 로직을 추가하세요.
    # 예를 들어, 세션이 이미 실행 중이라면 'ready' 상태를 emit합니다.
    if fiftyone_thread and fiftyone_thread.is_alive():
        emit('fiftyone_ready', {'status': 'ready'})
    else:
        emit('fiftyone_ready', {'status': 'not_ready'})

if __name__ == "__main__":

    socketio.run(app, port=5555, debug=False)