#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import tarfile

if os.environ.get('SYSTEM') == 'spaces':
    subprocess.call('pip uninstall -y opencv-python'.split())
    subprocess.call('pip uninstall -y opencv-python-headless'.split())
    subprocess.call('pip install opencv-python-headless==4.5.5.64'.split())

import gradio as gr
import huggingface_hub
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

TITLE = 'MediaPipe Face Mesh'
DESCRIPTION = 'https://google.github.io/mediapipe/'
ARTICLE = None

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    parser.add_argument('--allow-screenshot', action='store_true')
    return parser.parse_args()


def load_sample_images() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        image_dir.mkdir()
        dataset_repo = 'hysts/input-images'
        filenames = ['001.tar', '005.tar']
        for name in filenames:
            path = huggingface_hub.hf_hub_download(dataset_repo,
                                                   name,
                                                   repo_type='dataset',
                                                   use_auth_token=TOKEN)
            with tarfile.open(path) as f:
                f.extractall(image_dir.as_posix())
    return sorted(image_dir.rglob('*.jpg'))


def run(
    image: np.ndarray,
    max_num_faces: int,
    min_detection_confidence: float,
    show_tesselation: bool,
    show_contours: bool,
    show_irises: bool,
) -> np.ndarray:
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence) as face_mesh:
        results = face_mesh.process(image)

    res = image[:, :, ::-1].copy()
    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            if show_tesselation:
                mp_drawing.draw_landmarks(
                    image=res,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.
                    get_default_face_mesh_tesselation_style())
            if show_contours:
                mp_drawing.draw_landmarks(
                    image=res,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.
                    get_default_face_mesh_contours_style())
            if show_irises:
                mp_drawing.draw_landmarks(
                    image=res,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.
                    get_default_face_mesh_iris_connections_style())

    return res[:, :, ::-1]


def main():
    args = parse_args()

    image_paths = load_sample_images()
    examples = [[path.as_posix(), 5, 0.5, True, True, True]
                for path in image_paths]

    gr.Interface(
        run,
        [
            gr.inputs.Image(type='numpy', label='Input'),
            gr.inputs.Slider(
                0, 10, step=1, default=5, label='Max Number of Faces'),
            gr.inputs.Slider(0,
                             1,
                             step=0.05,
                             default=0.5,
                             label='Minimum Detection Confidence'),
            gr.inputs.Checkbox(default=True, label='Show Tesselation'),
            gr.inputs.Checkbox(default=True, label='Show Contours'),
            gr.inputs.Checkbox(default=True, label='Show Irises'),
        ],
        gr.outputs.Image(type='numpy', label='Output'),
        examples=examples,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_screenshot=args.allow_screenshot,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
