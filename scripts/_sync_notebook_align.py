"""
Script TAM THOI - dong bo notebook Colab (FADING_pipeline_colab_1.ipynb) voi thay doi da
lam o main.py: them Module 1.5 (align_to_ffhq()) - phat hien qua doi chieu thuc te, notebook
va VS Code la 2 codebase rieng biet, sua o 1 ben KHONG tu dong ap dung cho ben kia.

3 thay doi:
  1. FaceEmbedder (cell dinh nghia class, notebook dang la ban CU, thieu detect_faces()) ->
     them method detect_faces() (giong het src/search/embedding.py ban hien tai).
  2. Chen 1 cell MOI ngay sau do, dinh nghia align_to_ffhq()/_ffhq_quad() (giong het
     src/utils/ffhq_align.py).
  3. Sua cell vong lap danh gia FG-NET (co dong "z_T, null_t, attn_maps = inverter.invert(
     pair["source_img"], ...)") -> chen align truoc khi goi invert(), dung DUNG anh da align
     (khong phai anh goc) cho Module 2.

Chi chinh sua FADING_pipeline_colab_1.ipynb (notebook chua vong lap danh gia FG-NET) -
FADING_pipeline_colab.ipynb (ban rut gon, khong co phan danh gia FG-NET) KHONG can sua vi
khong lien quan.
"""

import json

NOTEBOOK_PATH = "notebooks/FADING_pipeline_colab_1.ipynb"

with open(NOTEBOOK_PATH, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ===== 1. Them detect_faces() vao FaceEmbedder (cell dinh nghia class) =====
FACE_EMBEDDER_MARKER = "class FaceEmbedder:"
target_idx = next(i for i, c in enumerate(cells) if FACE_EMBEDDER_MARKER in "".join(c.get("source", [])))
src_lines = cells[target_idx]["source"]
full_src = "".join(src_lines)
assert "detect_faces" not in full_src, "detect_faces() da co san trong notebook - khong can them lai"

DETECT_FACES_METHOD = '''
    def detect_faces(self, image_path: str) -> list:
        """Add-on cho buoc align (Module 1.5) - chay detection 1 lan, tra ve NGUYEN list
        doi tuong Face tho cua insightface (co .bbox, .kps 5 diem moc, .det_score,
        .normed_embedding). KHONG raise neu 0 mat, tra ve [] - nguoi goi tu quyet dinh cach
        xu ly (vd raise ValueError ro rang o noi goi, xem ham align_to_ffhq() ben duoi)."""
        if self.app is None:
            self._load_model()

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Khong doc duoc anh: {image_path}")

        return self.app.get(img)
'''

# Chen method ngay sau ham embed(), truoc build_gallery() - tim vi tri "def build_gallery"
build_gallery_line_idx = next(i for i, line in enumerate(src_lines) if "def build_gallery" in line)
new_src_lines = (
    src_lines[:build_gallery_line_idx]
    + [line + "\n" for line in DETECT_FACES_METHOD.strip("\n").split("\n")]
    + ["\n"]
    + src_lines[build_gallery_line_idx:]
)
cells[target_idx]["source"] = new_src_lines
cells[target_idx]["outputs"] = []
cells[target_idx]["execution_count"] = None
print(f"[1/3] Da them detect_faces() vao FaceEmbedder (cell {target_idx}).")

# ===== 2. Chen cell MOI dinh nghia align_to_ffhq() ngay sau cell FaceEmbedder =====
ALIGN_CELL_SOURCE = '''import cv2
import numpy as np


def _ffhq_quad(eye_left, eye_right, mouth_left, mouth_right):
    """Dung "quad" (hinh vuong xoay dinh vi vung align) - DUNG NGUYEN VAN cong thuc goc
    NVlabs/ffhq-dataset (download_ffhq.py, ham recreate_aligned_images), chi thay 4 diem dau
    vao tu dlib 68-diem sang insightface 5-diem (eye_left/eye_right = tam mat co san, khong
    can tinh trung binh contour; mouth_left/mouth_right = 2 khoe mieng)."""
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * np.array([-1, 1])
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * np.array([-1, 1])
    c0 = eye_avg + eye_to_mouth * 0.1

    # Thu tu dung PIL.Image.QUAD mong doi: tren-trai, duoi-trai, duoi-phai, tren-phai.
    return np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])


def align_to_ffhq(image_bgr: np.ndarray, kps: np.ndarray, output_size: int = 256) -> np.ndarray:
    """Align anh ve output_size x output_size theo DUNG cong thuc goc FFHQ (NVIDIA). `kps`:
    5 diem insightface (mat trai, mat phai, mui, khoe mieng trai, khoe mieng phai) - CHI dung
    4 diem mat+mieng, bo qua diem mui (cong thuc goc khong can). `quad` (4 diem) tao thanh 1
    hinh vuong xoay (x vuong goc y, |x|=|y|) nen map 3/4 goc bang cv2.getAffineTransform la
    du chinh xac tuyet doi.

    FIX (phat hien qua chan doan thuc te tren FG-NET, xem outputs/diag_fgnet_*_log.txt): dua
    thang anh CHUA align (anh "trong tu nhien") qua Module 2 cho ID Score gan 0 hoac AM -
    UNet fine-tune (Module 1) chi hoc tren bo cuc FFHQ-align, anh lech bo cuc lam sinh sai
    hoan toan cau truc khuon mat. BAT BUOC ap dung cho MOI anh input, khong chi anh FFHQ."""
    eye_left, eye_right = kps[0], kps[1]
    mouth_left, mouth_right = kps[3], kps[4]

    quad = _ffhq_quad(eye_left, eye_right, mouth_left, mouth_right).astype(np.float32)
    dst = np.array(
        [[0, 0], [0, output_size], [output_size, output_size], [output_size, 0]], dtype=np.float32
    )

    M = cv2.getAffineTransform(quad[:3], dst[:3])
    return cv2.warpAffine(image_bgr, M, (output_size, output_size), borderMode=cv2.BORDER_REFLECT)
'''

align_markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Module 1.5 — Align FFHQ-exact (tương đương `src/utils/ffhq_align.py`)\n",
        "\n",
        "**Thêm sau khi phát hiện qua chẩn đoán thực tế** (không có trong bản notebook gốc): "
        "SD1.5 UNet fine-tune ở Module 1 chỉ học trên bố cục FFHQ-align — ảnh input lệch bố "
        "cục (ảnh \"trong tự nhiên\" như FG-NET, chưa align) làm Module 2/3 sinh sai hoàn toàn "
        "cấu trúc khuôn mặt. BẮT BUỘC áp dụng cho mọi ảnh input trước khi vào Module 2.",
    ],
}
align_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in ALIGN_CELL_SOURCE.strip("\n").split("\n")],
}

insert_at = target_idx + 1
cells[insert_at:insert_at] = [align_markdown_cell, align_code_cell]
print(f"[2/3] Da chen cell markdown + code dinh nghia align_to_ffhq() tai vi tri {insert_at}.")

# ===== 3. Sua vong lap danh gia FG-NET - chen align truoc invert() =====
eval_loop_idx = next(
    i for i, c in enumerate(cells)
    if 'inverter.invert(pair["source_img"]' in "".join(c.get("source", []))
)
eval_src_lines = cells[eval_loop_idx]["source"]
eval_full_src = "".join(eval_src_lines)

OLD_LINE = '            z_T, null_t, attn_maps = inverter.invert(pair["source_img"], pair["source_age"], gender_word)\n'
NEW_LINES = [
    "            # Module 1.5 (align) - BAT BUOC tu sau Van de #1 (xem cell align_to_ffhq o tren):\n",
    "            # FG-NET la anh \"trong tu nhien\", CHUA align nhu FFHQ - phai align truoc khi vao\n",
    "            # Module 2, neu khong ID Score se gan 0 hoac AM (da do thuc te, xem outputs/diag_fgnet_no_align_log.txt).\n",
    "            source_faces = embedder.detect_faces(pair[\"source_img\"])\n",
    "            if len(source_faces) == 0:\n",
    "                raise ValueError(f\"Align that bai: khong phat hien duoc mat trong {pair['source_img']}\")\n",
    "            source_image_bgr = cv2.imread(pair[\"source_img\"])\n",
    "            aligned_source_path = os.path.join(OUTPUT_DIR, \"aligned_input.png\")\n",
    "            cv2.imwrite(aligned_source_path, align_to_ffhq(source_image_bgr, source_faces[0].kps, output_size=256))\n",
    "\n",
    "            z_T, null_t, attn_maps = inverter.invert(aligned_source_path, pair[\"source_age\"], gender_word)\n",
]

assert OLD_LINE in eval_src_lines, "Khong tim thay dong invert() can sua - kiem tra lai cau truc cell"
line_idx = eval_src_lines.index(OLD_LINE)
cells[eval_loop_idx]["source"] = eval_src_lines[:line_idx] + NEW_LINES + eval_src_lines[line_idx + 1:]
cells[eval_loop_idx]["outputs"] = []
cells[eval_loop_idx]["execution_count"] = None
print(f"[3/3] Da chen buoc align vao vong lap danh gia FG-NET (cell {eval_loop_idx}).")

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nHoan tat. Da luu lai notebook.")
