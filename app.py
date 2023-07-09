#!/usr/bin/env python

from __future__ import annotations

import pathlib

import gradio as gr
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

TITLE = 'MediaPipe Face Mesh'
DESCRIPTION = 'https://google.github.io/mediapipe/'


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


image_paths = sorted(pathlib.Path('images').rglob('*.jpg'))
examples = [[path, 5, 0.5, True, True, True] for path in image_paths]

gr.Interface(
    fn=run,
    inputs=[
        gr.Image(label='Input', type='numpy'),
        gr.Slider(label='Max Number of Faces',
                  minimum=0,
                  maximum=10,
                  step=1,
                  value=5),
        gr.Slider(label='Minimum Detection Confidence',
                  minimum=0,
                  maximum=1,
                  step=0.05,
                  value=0.5),
        gr.Checkbox(label='Show Tesselation', value=True),
        gr.Checkbox(label='Show Contours', value=True),
        gr.Checkbox(label='Show Irises', value=True),
    ],
    outputs=gr.Image(label='Output', height=500),
    examples=examples,
    title=TITLE,
    description=DESCRIPTION,
).queue().launch()
