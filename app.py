import fiftyone as fo
import fiftyone.brain as fob
import clip
import torch

import numpy as np

from flask import Flask, render_template, request, redirect, url_for, Response, send_file, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

import os
import zipfile
import sys
import argparse
from tqdm import tqdm
import threading
import io
import time
import subprocess
import atexit
import json
from tqdm import tqdm
import shutil

from trainer import train_yolo
from utils import TensorboardManager, FiftyoneManager, CaptureOutput, InputDataLoader

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_dir", type=str, required=True, help="Importing dataset path")
parser.add_argument("--dataset_name", type=str, default="imported_dataset", help="Name the dataset you are importing")
parser.add_argument("--port", type=int, default=5151, help="Port to run the FiftyOne app on")
parser.add_argument("--dataset_type", type=str, default=None, help="dataset type (51, yolo)")

args = parser.parse_args()

# 데이터셋 로드 및 세션 생성은 애플리케이션 시작 시 한 번만 수행
tsb_runner = TensorboardManager(port=6006)
atexit.register(tsb_runner.stop)

# ffto_runner = FiftyoneManager(args.dataset_dir, args.dataset_name, args.dataset_type, args.port)
# fiftyone_thread, dataset, dataset_type = ffto_runner.start()

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# ffto_runner.emit_event(socketio, 'fiftyone_ready', {'status': 'ready'})

## logger instance
capture_stream = CaptureOutput()
sys.stdout = capture_stream

@app.route('/')
def index():
    return redirect(url_for('init_data'))

# @app.route('/get_views', methods=['GET'])
# def get_views():
#     # Fetch the list of saved views
#     list_views = dataset.list_saved_views()
#     return {'views': list_views}

@app.route('/init_data')
def init_data():
    return render_template('init_data.html')

@app.route('/upload', methods=['POST'])
def upload_file():

    UPLOAD_FOLDER = './datasets/uploads'
    dataset_infos = {}

    print("Starting file upload process...")  # 디버그 출력
    for key in request.files:
        file = request.files[key]
        selected_format = request.form.get(f"{key.split('-')[0]}-format")
        print(f"Processing file: {file.filename}, Format: {selected_format}")  # 디버그 출력

        # if file.filename == '':
        #     return jsonify({'message': 'No selected file'}), 400
        
        if file and file.filename.endswith('.zip'):
            zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(zip_path)
            print(f"File saved to: {zip_path}")  # 디버그 출력

            # Extract the ZIP file
            try:
                if os.path.exists(os.path.splitext(zip_path)[0]):
                    print(f"Removing existing directory: {os.path.splitext(zip_path)[0]}")
                    shutil.rmtree(os.path.splitext(zip_path)[0])
                    print(f"Removing existing dataset: {os.path.splitext(file.filename)[0]}")
                    fo.delete_dataset(os.path.splitext(file.filename)[0])
                else:
                    print(f"No existing directory found: {os.path.splitext(zip_path)[0]}")

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for file in tqdm(zip_ref.namelist(), desc="Extracting files"):
                        zip_ref.extract(file, UPLOAD_FOLDER)

                print(f"Extracted ZIP file: {zip_path}")  # 디버그 출력

            except Exception as e:
                print(f"Error extracting ZIP file {zip_path}: {e}")

            # Remove the original ZIP file
            try:
                os.remove(zip_path)
                print(f"Removed ZIP file: {zip_path}")  # 디버그 출력

            except Exception as e:
                print(f"Error removing file {zip_path}: {e}")

            data_path = os.path.splitext(zip_path)[0]
            print(data_path)
            loader = InputDataLoader(data_path, selected_format)
            
            if key == "ref-upload":
                ref_dataset = loader.get_img_data()
                loader.add_tags("ref")
                dataset_infos['ref'] = loader.get_dataset_info()
            elif key == "cur-upload":
                cur_dataset = loader.get_img_data()
                loader.add_tags("cur")
                dataset_infos['cur'] = loader.get_dataset_info()
            elif key == "test-upload":
                test_dataset = loader.get_img_data()
                loader.add_tags("test")
                dataset_infos['test'] = loader.get_dataset_info()

    merged_dataset_name = request.form.get('merged-dataset-name')
    fiftyone_port_number = request.form.get('port-number')

    # 기존 데이터셋 삭제
    if fo.dataset_exists(merged_dataset_name):
        print(f"Deleting existing dataset: {merged_dataset_name}")
        fo.delete_dataset(merged_dataset_name)

    print("Loading Embedding Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/16", device=device)

    print(f"Merged Dataset Name: {merged_dataset_name}, Port Number: {fiftyone_port_number}")  # 디버그 출력
    merged_dataset = fo.Dataset(merged_dataset_name)
    merged_dataset.add_samples(ref_dataset)
    merged_dataset.add_samples(cur_dataset)
    merged_dataset.add_samples(test_dataset)

    print("Calculating Embeddings...")
    loader.get_embeddings(merged_dataset, device, model, preprocess)
    embeddings_by_sample_id = loader.collect_image_embeddings_by_sample_id(merged_dataset)
    embedding_vis_result = fob.compute_visualization(
        merged_dataset,
        embeddings=embeddings_by_sample_id,
        brain_key="clip_embeddings",
        plot_points=True,
        verbose=True,
    )
    
    if dataset_infos:
        print("Successfully processed all datasets.")  # 디버그 출력
        # /dataclinic 페이지로 리다이렉트하며 데이터 전달
        return redirect(url_for(
            'dataclinic',
            merged_dataset_name=merged_dataset_name,
            vis_result=embedding_vis_result,
            fiftyone_port_number=fiftyone_port_number,
            data_infos=json.dumps(dataset_infos)
        ))
    else:
        print("No valid files processed.")  # 디버그 출력
        return jsonify({'message': 'Invalid file type'}), 400

@app.route('/dataclinic')
def dataclinic():
    # merged_dataset과 vis_result를 가져옵니다.
    merged_dataset_name = request.args.get('merged_dataset_name')
    vis_result = request.args.get('vis_result')
    fiftyone_port_number = request.args.get('fiftyone_port_number')
    data_infos = request.args.get('data_infos')

    # list_views를 가져옵니다.
    list_views = []  # 실제 데이터셋에서 뷰 목록을 가져오는 로직을 추가하세요.
    if merged_dataset_name:
        dataset = fo.load_dataset(merged_dataset_name)
        list_views = dataset.list_saved_views()
    
    return render_template(
        'dataclinic.html',
        list_views=list_views,
        port=fiftyone_port_number,
        merged_dataset_name=merged_dataset_name,
        vis_result=vis_result,
        data_infos=data_infos
        )

# # @app.route('/save', methods=['POST'])
# # def save_dataset():
# #     # 데이터셋 변경사항 저장 및 내보내기
# #     print()
# #     print("Save Changes of Dataset...")
# #     print()
# #     # export_dir = args.dataset_dir
# #     # dataset.export(
# #     #     export_dir=export_dir,
# #     #     dataset_type=fo.types.FiftyOneDataset,
# #     # )
# #     dataset.save()

# #     return redirect(url_for('home'))

# @app.route('/export', methods=['POST'])
# def export_selected_view():
#     # 선택된 뷰 가져오기
#     selected_view = request.form.get('selected_view')
#     selected_format = request.form.get('selected_format')
#     print(f"Selected view: {selected_view}, Selected format: {selected_format}")  # 디버깅을 위한 출력

#     if selected_view and selected_format:
#         print(f"Exporting View: {selected_view}")
#         view = dataset.load_saved_view(selected_view)
#         view_export_dir = f"./datasets/exported_datasets/{dataset.name}_{selected_view}"
#         label_field = "ground_truth"

#         # 전체 샘플을 train, val로 스플릿
#         splits = ['train', 'val']
#         split_ratios = [0.8, 0.2]  # 예시 비율
#         view.shuffle(seed=42)  # 랜덤 시드로 셔플

#         # 기존의 train, val, test 태그 삭제 및 새로 추가
#         num_samples = len(view)
#         split_indices = np.cumsum([int(r * num_samples) for r in split_ratios])

#         for idx, sample in enumerate(view):
#             # 기존 태그 중 train, val, test 제거
#             sample.tags = [tag for tag in sample.tags if tag not in splits]

#             # 새로운 스플릿 태그 추가
#             if idx < split_indices[0]:
#                 sample.tags.append('train')
#             elif idx < split_indices[1]:
#                 sample.tags.append('val')
#             sample.save()

#         # 내보내기 포맷 선택
#         if selected_format == "YOLOv5Dataset":
#             dataset_type = fo.types.YOLOv5Dataset
#         elif selected_format == "FiftyOneDataset":
#             dataset_type = fo.types.FiftyOneDataset

#         # 내보내기
#         for split in splits:
#             split_view = view.match_tags(split)
#             split_view.export(
#                 export_dir=view_export_dir,
#                 dataset_type=dataset_type,
#                 label_field=label_field,
#                 split=split,
#             )

#         print(f"Exported to {view_export_dir}")

#     return redirect(url_for('home'))

# @app.route('/train_page', methods=['GET', 'POST'])
# def train_page():
#     # Exported datasets directory
#     export_dir = './datasets/exported_datasets/'
#     # Models directory
#     models_dir = './models/'

#     # 데이터셋과 모델의 경로를 저장
#     datasets = [os.path.join(export_dir, d) for d in os.listdir(export_dir) if os.path.isdir(os.path.join(export_dir, d))]
#     models = [os.path.join(models_dir, m) for m in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, m))]

#     return render_template('train_page.html', datasets=datasets, models=models)

# @app.route('/train', methods=['POST'])
# def train():

#     selected_dataset = request.form.get('selected_dataset') + '/dataset.yaml'
#     selected_model = request.form.get('selected_model')

#     # 별도의 스레드에서 훈련 시작
#     training_thread = threading.Thread(target=train_yolo, args=(selected_dataset, selected_model))
#     training_thread.start()

#     tensorboard_thread = threading.Thread(target=tsb_runner.start)
#     tensorboard_thread.start()
#     tsb_runner.emit_event(socketio, 'tensorboard_ready', {'status': 'ready'})

#     return redirect(url_for('train_page'))

# @app.route('/download_model')
# def download_model():
#     model_path = 'runs/train/weights/best.pt'
#     return send_file(model_path, as_attachment=True)

# @app.route('/stream_logs')
# def stream_logs():
#     def generate():
#         while True:
#             line = capture_stream.get_output()
#             if line:
#                 yield f"data: {line}\n\n"
#                 capture_stream.clear_output() # 로그 전송 후 초기화
#             time.sleep(1)  # Add a small delay to prevent high CPU usage

#     return Response(generate(), mimetype='text/event-stream')

# @socketio.on('check_fiftyone_ready')
# def handle_check_fiftyone_ready():
#     # FiftyOne 세션이 준비되었는지 확인하는 로직을 추가하세요.
#     # 예를 들어, 세션이 이미 실행 중이라면 'ready' 상태를 emit합니다.
#     if fiftyone_thread and fiftyone_thread.is_alive():
#         emit('fiftyone_ready', {'status': 'ready'})
#     else:
#         emit('fiftyone_ready', {'status': 'not_ready'})

if __name__ == "__main__":

    socketio.run(app, port=5555, debug=False)