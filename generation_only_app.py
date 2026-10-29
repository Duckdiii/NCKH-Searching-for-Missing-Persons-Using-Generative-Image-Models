"""Standalone FADING image editing UI; no identity search or camera integration."""
import gc
import os
from pathlib import Path
from uuid import uuid4

import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'checkpoints/specialized_unet_v3_v2_600samples/specialized_unet_v3/best'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

st.set_page_config(page_title='FADING — chỉ sinh ảnh', layout='wide')
st.title('FADING — chỉ sinh ảnh')
st.info('Không truy xuất danh tính, không kết nối gallery hoặc camera. Ảnh sinh là minh họa, không phải dự đoán chính xác diện mạo.')
st.code(str(CHECKPOINT))
if not (CHECKPOINT / 'diffusion_pytorch_model.safetensors').is_file():
    st.error('Thiếu trọng số checkpoint. Không tự chuyển sang model khác.')
    st.stop()
st.caption('Đã chọn checkpoint best. Model chỉ nạp khi bấm Sinh ảnh; không tải model qua mạng.')
upload = st.file_uploader('Ảnh chân dung đã crop/căn chỉnh (bố cục FFHQ)', type=['png', 'jpg', 'jpeg'])
initial_age = st.number_input('Tuổi trong ảnh', 1, 100, 20)
target_age = st.number_input('Tuổi minh họa mong muốn', 1, 100, 30)
gender = st.selectbox('Từ mô tả trong prompt', ['person', 'man', 'woman'])
ready = st.checkbox('Ảnh đã được crop/căn chỉnh; tôi có quyền sử dụng ảnh này.')
if upload:
    st.image(upload, caption='Ảnh đầu vào', width=320)
if st.button('Sinh ảnh', disabled=not (upload and ready)):
    inverter = editor = None
    try:
        import torch
        from src.fading.inversion import NullTextInverter
        from src.fading.editing import Editor
        if not torch.cuda.is_available():
            raise RuntimeError('Môi trường hiện tại chưa có CUDA khả dụng; chưa chạy sinh ảnh.')
        job = ROOT / 'outputs/generation_only' / uuid4().hex
        job.mkdir(parents=True)
        source = job / 'input.png'
        ImageOps.exif_transpose(Image.open(upload)).convert('RGB').save(source)
        common = dict(unet_checkpoint_dir=str(CHECKPOINT), device='cuda', num_inference_steps=50)
        with st.spinner('Đang inversion và sinh ảnh; không chạy truy xuất danh tính...'):
            inverter = NullTextInverter(**common)
            z, nulls, maps = inverter.invert(str(source), int(initial_age), gender)
            del inverter
            inverter = None
            gc.collect()
            torch.cuda.empty_cache()
            editor = Editor(**common)
            results = editor.edit(z, nulls, maps, [int(target_age)], gender, str(job))
            st.session_state['generation_result'] = str(results[int(target_age)])
    except Exception as exc:
        st.error(f'Không sinh được ảnh: {exc}')
    finally:
        del inverter, editor
        gc.collect()
        if 'torch' in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
if st.session_state.get('generation_result'):
    st.image(st.session_state['generation_result'], caption='Ảnh minh họa do model sinh', width=400)
