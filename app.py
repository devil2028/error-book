# -*- coding: utf-8 -*-
"""
错题白卷生成器（Windows 本地版 · 拖框选择）
流程：上传试卷照片 → 拖动鼠标框选错题 → OCR+楷体重排（带图则拆分配图擦除）→ 拼成 A4 白卷 PDF
"""
import json
import os
import re
import time
import traceback
import uuid
from urllib.parse import quote

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps
from flask import Flask, jsonify, render_template, request, send_from_directory

from typeset import assess_erased_question, process_question

# ---------- 配置 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
LOCAL_CONFIG_PATH = os.path.join(BASE_DIR, "config.local.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 默认按 150dpi 生成 A4。手机截图和聊天软件转发图常只有 600px 左右宽，
# 以 300dpi 排版会把单题放大近三倍，字边缘反而更糊。
A4_W, A4_H = 1240, 1754          # A4 @150dpi
MARGIN = 60                       # 页边距
GAP = 40                         # 多列时列间距
COLS = 1                         # 每行几道题（单列：题目最大最清晰，适合错题重练）
MAX_SIDE = 4000                  # 发给接口的单题图长边上限
MIN_API_SIDE = 800               # 仅当长边低于接口建议下限时才轻量放大；过高会先造糊再擦
JPEG_QUALITY = 95                # 预览 JPG / PDF 内嵌图质量
DEFAULT_DRAFT_GAP = 100          # 题与题之间预留草稿行距（像素 @150dpi）

MODE_LABEL = {
    "image_clean": "保真擦除",
    "text_only": "文字",
    "mixed": "带图",
    "fallback": "擦除兜底",
}

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024  # 多图上传上限 80MB

_client = None
_config = None


def job_dir():
    """返回与当前输出目录绑定的预览任务存储目录。"""
    return os.path.join(OUTPUT_DIR, "jobs")


def load_config():
    """读取 config.json，返回 dict。"""
    global _config
    if _config is not None:
        return _config
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        local = {}
        if os.path.isfile(LOCAL_CONFIG_PATH):
            with open(LOCAL_CONFIG_PATH, encoding="utf-8") as f:
                local = json.load(f)
        sid = str(os.environ.get("TENCENTCLOUD_SECRET_ID", local.get("secret_id", ""))).strip()
        skey = str(os.environ.get("TENCENTCLOUD_SECRET_KEY", local.get("secret_key", ""))).strip()
        if not sid or sid.startswith("请填写") or not skey or skey.startswith("请填写"):
            _config = {"ok": False, "reason": "请设置 TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY 或 config.local.json"}
        else:
            _config = {
                "ok": True,
                "secret_id": sid,
                "secret_key": skey,
                "region": cfg.get("region", "ap-guangzhou"),
                "crop": int(cfg.get("crop", 0)),
                "deskew": int(cfg.get("deskew", 0)),
                "sharpen": int(cfg.get("sharpen", 0)),
                "grayscale": int(cfg.get("grayscale", 1)),
                "quality_check": int(cfg.get("quality_check", 1)),
                "mode": str(cfg.get("mode", "hybrid")),
                "kaiti_font": str(cfg.get("kaiti_font", "")),
                "font_size": int(cfg.get("font_size", 38)),
                "block_gap": int(cfg.get("block_gap", 24)),
                "draft_gap": int(cfg.get("draft_gap", DEFAULT_DRAFT_GAP)),
                "draft_lines": int(cfg.get("draft_lines", 3)),
                "ocr_min_avg_conf": float(cfg.get("ocr_min_avg_conf", 60)),
                "ocr_line_min_conf": float(cfg.get("ocr_line_min_conf", 55)),
                "ocr_min_lines": int(cfg.get("ocr_min_lines", 1)),
                "text_coverage_high": float(cfg.get("text_coverage_high", 0.12)),
                "text_coverage_low": float(cfg.get("text_coverage_low", 0.04)),
                "figure_gap_ratio": float(cfg.get("figure_gap_ratio", 0.18)),
                "figure_min_h_ratio": float(cfg.get("figure_min_h_ratio", 0.18)),
                "figure_min_w_ratio": float(cfg.get("figure_min_w_ratio", 0.25)),
                "figure_min_area_ratio": float(cfg.get("figure_min_area_ratio", 0.08)),
            }
    except Exception as e:
        _config = {"ok": False, "reason": f"config.json 读取失败：{e}"}
    return _config


def get_client():
    """懒加载腾讯云 OCR 客户端。"""
    global _client
    if _client is not None:
        return _client
    cfg = load_config()
    if not cfg["ok"]:
        raise RuntimeError(cfg["reason"])
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.ocr.v20181119 import ocr_client

    cred = credential.Credential(cfg["secret_id"], cfg["secret_key"])
    http_profile = HttpProfile()
    http_profile.endpoint = "ocr.tencentcloudapi.com"
    client_profile = ClientProfile()
    client_profile.httpProfile = http_profile
    _client = ocr_client.OcrClient(cred, cfg["region"], client_profile)
    return _client


def prepare_crop_for_api(img: Image.Image) -> Image.Image:
    """送接口前：尽量保持原裁剪分辨率；仅过小才轻量放大，过大才缩小。"""
    w, h = img.size
    long_side = max(w, h)
    if long_side < MIN_API_SIDE:
        scale = MIN_API_SIDE / long_side
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.LANCZOS,
        )
    if max(img.size) > MAX_SIDE:
        img = img.copy()
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img


def cleanup_colored_annotations(original: Image.Image, erased: Image.Image) -> Image.Image:
    """利用原图颜色定位红笔批注，清掉云端结果中残留的同一区域。

    只处理红色明显强于绿色和蓝色的像素，避免伤及黑色印刷题干和配图；
    掩膜略微膨胀，以覆盖擦除接口留下的灰黑边缘。
    """
    target = erased.convert("RGB")
    source = original.convert("RGB").resize(target.size, Image.LANCZOS)
    red, green, blue = source.split()
    red_over_green = ImageChops.subtract(red, green)
    red_over_blue = ImageChops.subtract(red, blue)
    red_dominance = ImageChops.darker(red_over_green, red_over_blue)
    mask = red_dominance.point(lambda value: 255 if value >= 24 else 0)
    mask = mask.filter(ImageFilter.MaxFilter(7))
    target.paste(Image.new("RGB", target.size, "white"), mask=mask)
    return target


def apply_manual_erase_masks(img: Image.Image, masks, crop_rect) -> Image.Image:
    """把用户在原图上标记的答案区域擦白，处理黑色笔迹残留。"""
    if not masks:
        return img
    x1, y1, x2, y2 = crop_rect
    crop_w, crop_h = x2 - x1, y2 - y1
    if crop_w <= 0 or crop_h <= 0:
        return img
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    sx, sy = out.width / crop_w, out.height / crop_h
    for mask in masks:
        try:
            if isinstance(mask, dict):
                values = [mask["x1"], mask["y1"], mask["x2"], mask["y2"]]
            else:
                values = mask
            mx1, my1, mx2, my2 = [int(float(value)) for value in values]
        except (TypeError, ValueError, KeyError):
            continue
        mx1, mx2 = sorted((max(0, mx1), min(x2, mx2)))
        my1, my2 = sorted((max(0, my1), min(y2, my2)))
        if mx2 <= mx1 or my2 <= my1:
            continue
        ix1, iy1 = max(x1, mx1), max(y1, my1)
        ix2, iy2 = min(x2, mx2), min(y2, my2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        pad_x = max(2, int(sx * 3))
        pad_y = max(2, int(sy * 3))
        draw.rectangle(
            (
                max(0, int((ix1 - x1) * sx) - pad_x),
                max(0, int((iy1 - y1) * sy) - pad_y),
                min(out.width, int((ix2 - x1) * sx) + pad_x),
                min(out.height, int((iy2 - y1) * sy) + pad_y),
            ),
            fill="white",
        )
    return out


def erase_one(img: Image.Image, cfg: dict) -> Image.Image:
    """把单张图片送去腾讯云擦除手写，返回擦除后的图片。"""
    import base64
    import io
    from tencentcloud.ocr.v20181119 import models

    def to_base64(im, fmt="PNG"):
        buf = io.BytesIO()
        rgb = im.convert("RGB")
        if fmt == "JPEG":
            rgb.save(buf, format="JPEG", quality=JPEG_QUALITY)
        else:
            rgb.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    b64 = to_base64(img, "PNG")
    if len(b64) > 7 * 1024 * 1024:
        img = img.copy()
        img.thumbnail((3000, 3000), Image.LANCZOS)
        b64 = to_base64(img, "PNG")
    if len(b64) > 7 * 1024 * 1024:
        b64 = to_base64(img, "JPEG")

    req = models.EraseHandwrittenImageOCRRequest()
    req.ImageBase64 = b64
    req.Crop = cfg["crop"]
    req.Deskew = cfg["deskew"]
    req.Sharpen = cfg["sharpen"]
    req.Grayscale = cfg.get("grayscale", 1)

    resp = get_client().EraseHandwrittenImageOCR(req)
    out = base64.b64decode(resp.Image)
    return Image.open(io.BytesIO(out)).convert("RGB")


def compose_a4_pages(clean_images, sharpen_erase=False, draft_gap=DEFAULT_DRAFT_GAP):
    """把多道题拼到若干张 A4 画布上（超出自动换页）。
    题与题之间留 draft_gap 空白，方便写草稿。
    返回 List[Image]。"""
    from PIL import ImageDraw

    pages = []
    canvas = Image.new("RGB", (A4_W, A4_H), "white")
    usable_w = A4_W - MARGIN * 2
    usable_h = A4_H - MARGIN * 2
    col_w = (usable_w - GAP * (COLS - 1)) // COLS
    x = MARGIN
    y = MARGIN
    max_h_in_row = 0
    n = len(clean_images)

    def new_page():
        nonlocal canvas, x, y, max_h_in_row
        pages.append(canvas)
        canvas = Image.new("RGB", (A4_W, A4_H), "white")
        x = MARGIN
        y = MARGIN
        max_h_in_row = 0

    for i, item in enumerate(clean_images):
        if isinstance(item, tuple):
            im, mode = item[:2]
            reference_page_width = item[2] if len(item) >= 3 else im.width
        else:
            im, mode = item, "fallback"
            reference_page_width = im.width
        w, h = im.size
        # 不把半页或窄栏框选硬拉成整行。以原始整页的宽度为基准，
        # 保留题目在试卷上的相对字号和图形比例。
        scale = min(col_w / w, col_w / max(1, reference_page_width))
        max_q_h = usable_h - (draft_gap if i < n - 1 else 0)
        # 单题过高时仍缩放到一页可用高度
        max_q_h = min(max_q_h, usable_h)
        if h * scale > max_q_h:
            scale = max_q_h / h
        if abs(scale - 1.0) > 0.01:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        if sharpen_erase and mode in ("fallback", "image_clean"):
            im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=5))
        w, h = im.size
        block_h = h + (draft_gap if i < n - 1 else 0)

        # 按实际题块宽度紧凑排版；窄栏题可与下一题同排，避免整页空白。
        if i > 0 and x + w > A4_W - MARGIN:
            x = MARGIN
            y += max_h_in_row + GAP
            max_h_in_row = 0

        if y + h > A4_H - MARGIN:
            new_page()
            x = MARGIN
            y = MARGIN
            max_h_in_row = 0

        canvas.paste(im, (x, y))
        if i < n - 1 and draft_gap >= 40:
            draw = ImageDraw.Draw(canvas)
            ly = y + h + 8
            for px in range(x, x + w, 12):
                draw.line([(px, ly), (min(px + 6, x + w), ly)], fill=(210, 210, 210), width=1)
        x += w + GAP
        max_h_in_row = max(max_h_in_row, block_h)

    pages.append(canvas)
    return pages


def compose_a4(clean_images, sharpen_erase=False, draft_gap=DEFAULT_DRAFT_GAP):
    """兼容旧调用：返回第一页（多页时请用 compose_a4_pages）。"""
    pages = compose_a4_pages(clean_images, sharpen_erase=sharpen_erase, draft_gap=draft_gap)
    return pages[0]


def stitch_preview(pages):
    """把多页竖向拼成一张预览图。"""
    if not pages:
        return Image.new("RGB", (A4_W, A4_H), "white")
    if len(pages) == 1:
        return pages[0]
    gap = 24
    total_h = sum(p.height for p in pages) + gap * (len(pages) - 1)
    out = Image.new("RGB", (A4_W, total_h), (245, 245, 245))
    y = 0
    for i, p in enumerate(pages):
        out.paste(p, (0, y))
        y += p.height + (gap if i < len(pages) - 1 else 0)
    return out


def save_result_pages(page_imgs, stamp, job_id=None):
    """保存 PDF、预览和可供预览后人工擦除的单页 PNG。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_name = f"错题白卷_{stamp}.pdf"
    jpg_name = f"错题白卷_{stamp}.jpg"
    pdf_path = os.path.join(OUTPUT_DIR, pdf_name)
    if len(page_imgs) == 1:
        page_imgs[0].save(pdf_path, "PDF", resolution=150, quality=JPEG_QUALITY, optimize=True)
    else:
        page_imgs[0].save(
            pdf_path, "PDF", resolution=150, quality=JPEG_QUALITY, optimize=True,
            save_all=True, append_images=page_imgs[1:],
        )
    stitch_preview(page_imgs).save(
        os.path.join(OUTPUT_DIR, jpg_name), "JPEG", quality=JPEG_QUALITY
    )
    if job_id:
        folder = os.path.join(job_dir(), job_id)
        os.makedirs(folder, exist_ok=True)
        for index, page in enumerate(page_imgs):
            page.save(os.path.join(folder, f"page_{index}.png"), "PNG", optimize=True)
        with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump({"page_count": len(page_imgs)}, handle)
    return pdf_name, jpg_name


def clip_box(box, w, h):
    """校验并裁剪框坐标为合法范围，返回 (x1,y1,x2,y2) 或 None。"""
    try:
        if isinstance(box, dict):
            x1 = float(box["x1"]); y1 = float(box["y1"])
            x2 = float(box["x2"]); y2 = float(box["y2"])
        else:
            x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 - x1 >= 20 and y2 - y1 >= 20:
        return (x1, y1, x2, y2)
    return None


def inset_box(rect, w, h, ratio=0.015, min_px=3, max_px=16):
    """框选略向内收缩，减少裁到邻题 / 栏外文字。"""
    x1, y1, x2, y2 = rect
    bw, bh = x2 - x1, y2 - y1
    if bw < 60 or bh < 40:
        return rect
    dx = min(max_px, max(min_px, int(bw * ratio)))
    dy = min(max_px, max(min_px, int(bh * ratio)))
    nx1, ny1 = x1 + dx, y1 + dy
    nx2, ny2 = x2 - dx, y2 - dy
    if nx2 - nx1 < 40 or ny2 - ny1 < 30:
        return rect
    nx1, ny1 = max(0, nx1), max(0, ny1)
    nx2, ny2 = min(w, nx2), min(h, ny2)
    return (nx1, ny1, nx2, ny2)


def column_width():
    usable_w = A4_W - MARGIN * 2
    return (usable_w - GAP * (COLS - 1)) // COLS


# ---------- 路由 ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/debug")
def debug():
    cfg = load_config()
    return jsonify(
        COLS=COLS,
        MAX_SIDE=MAX_SIDE,
        MIN_API_SIDE=MIN_API_SIDE,
        JPEG_QUALITY=JPEG_QUALITY,
        A4_W=A4_W,
        A4_H=A4_H,
        mode=cfg.get("mode"),
        crop=cfg.get("crop"),
        deskew=cfg.get("deskew"),
        sharpen=cfg.get("sharpen"),
        grayscale=cfg.get("grayscale"),
        font_size=cfg.get("font_size"),
    )


@app.route("/output/<path:name>")
def output_file(name):
    return send_from_directory(OUTPUT_DIR, name)


@app.route("/api/erase", methods=["POST"])
def api_erase():
    cfg = load_config()
    if not cfg["ok"]:
        return jsonify(ok=False, error=cfg["reason"])

    # 兼容单图 file / 多图 files
    files = request.files.getlist("files")
    if not files:
        one = request.files.get("file")
        if one is not None:
            files = [one]
    files = [f for f in files if f and getattr(f, "filename", None)]
    if not files:
        return jsonify(ok=False, error="未收到图片，请重新上传")

    # pages: [{boxes:[...]}, ...] 与 files 一一对应；旧版仅传 boxes 视为单页
    pages_meta = None
    raw_pages = request.form.get("pages")
    if raw_pages:
        try:
            pages_meta = json.loads(raw_pages)
        except Exception:
            return jsonify(ok=False, error="多页框选数据解析失败")
    if pages_meta is None:
        try:
            raw_boxes = json.loads(request.form.get("boxes", "[]"))
        except Exception:
            return jsonify(ok=False, error="框选坐标解析失败")
        pages_meta = [{"boxes": raw_boxes}]

    if len(pages_meta) != len(files):
        return jsonify(
            ok=False,
            error=f"图片数量({len(files)})与框选页数({len(pages_meta)})不一致",
        )

    try:
        client = get_client()
        col_w = column_width()
        pipeline = cfg.get("mode", "hybrid")
        results = []
        modes = []
        review_warnings = []
        page_counts = []

        for fi, f in enumerate(files):
            raw_boxes = pages_meta[fi].get("boxes") if isinstance(pages_meta[fi], dict) else pages_meta[fi]
            raw_masks = pages_meta[fi].get("erase_masks", []) if isinstance(pages_meta[fi], dict) else []
            if not raw_boxes:
                page_counts.append(0)
                continue
            src = Image.open(f.stream)
            src = ImageOps.exif_transpose(src).convert("RGB")
            w, h = src.size
            rects = [r for r in (clip_box(b, w, h) for b in raw_boxes) if r]
            page_counts.append(len(rects))
            for qi, rect in enumerate(rects, 1):
                x1, y1, x2, y2 = inset_box(rect, w, h)
                crop = src.crop((x1, y1, x2, y2))
                try:
                    if pipeline == "erase":
                        prepared = prepare_crop_for_api(crop)
                        img = cleanup_colored_annotations(
                            prepared, erase_one(prepared, cfg)
                        )
                        mode = "image_clean"
                    else:
                        def _erase(im, c):
                            prepared = prepare_crop_for_api(im)
                            erased = erase_one(prepared, c)
                            return cleanup_colored_annotations(prepared, erased)
    
                        img, mode = process_question(
                            crop, cfg, client, _erase, col_w, JPEG_QUALITY
                        )
                except Exception:
                    app.logger.exception("题目处理失败：图片 %s，框选 %s", fi + 1, qi)
                    return jsonify(ok=False, error=f"第 {fi + 1} 张图第 {qi} 个框选处理失败，未生成白卷。请检查云服务配置和网络后重试。", failed_page=fi + 1, failed_box=qi), 502
                # API 可能对小图放大或对大图缩小；换算出处理后图像中
                # 对应的“整页宽度”，供 A4 排版保持原始相对比例。
                reference_page_width = max(1, round(src.width * img.width / crop.width))
                img = apply_manual_erase_masks(img, raw_masks, (x1, y1, x2, y2))
                results.append((img, mode, reference_page_width))
                modes.append(mode)
                warnings = []
                if mode == "image_clean" and int(cfg.get("quality_check", 1)):
                    warnings = assess_erased_question(client, img, JPEG_QUALITY)
                if warnings:
                    review_warnings.append({
                        "page": fi + 1,
                        "selection": qi,
                        "messages": warnings,
                    })

        if not results:
            return jsonify(ok=False, error="请先在各张图上拖动鼠标框选错题")

        page_imgs = compose_a4_pages(
            results,
            sharpen_erase=True,
            draft_gap=int(cfg.get("draft_gap", DEFAULT_DRAFT_GAP)),
        )
        stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex
        job_id = uuid.uuid4().hex
        pdf_name, jpg_name = save_result_pages(page_imgs, stamp, job_id)

        mode_summary = "、".join(
            f"{i+1}:{MODE_LABEL.get(m, m)}" for i, m in enumerate(modes)
        )
        return jsonify(
            ok=True,
            count=len(results),
            selection_count=len(results),
            page_count=len(files),
            pdf_pages=len(page_imgs),
            page_counts=page_counts,
            modes=modes,
            mode_labels=[MODE_LABEL.get(m, m) for m in modes],
            mode_summary=mode_summary,
            needs_review=bool(review_warnings),
            review_warnings=review_warnings,
            pdf="/output/" + quote(pdf_name),
            preview="/output/" + quote(jpg_name),
            job_id=job_id,
            preview_page_height=A4_H,
            preview_gap=24,
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=f"处理失败：{e}")


@app.route("/api/manual-erase-result", methods=["POST"])
def api_manual_erase_result():
    """在生成后的预览上擦除残留笔迹，无需重新上传或调用云端。"""
    data = request.get_json(silent=True) or {}
    job_id = str(data.get("job_id", ""))
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        return jsonify(ok=False, error="预览任务已失效，请重新生成白卷"), 400
    folder = os.path.join(job_dir(), job_id)
    manifest_path = os.path.join(folder, "manifest.json")
    if not os.path.isfile(manifest_path):
        return jsonify(ok=False, error="预览任务已失效，请重新生成白卷"), 404
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            page_count = int(json.load(handle)["page_count"])
        masks = data.get("masks", [])
        if not isinstance(masks, list) or len(masks) > 200:
            return jsonify(ok=False, error="擦除标记数量无效"), 400
        pages = []
        for page_index in range(page_count):
            page_path = os.path.join(folder, f"page_{page_index}.png")
            page = Image.open(page_path).convert("RGB")
            page_masks = [mask for mask in masks if isinstance(mask, dict) and mask.get("page") == page_index]
            pages.append(apply_manual_erase_masks(page, page_masks, (0, 0, page.width, page.height)))
        stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex
        next_job_id = uuid.uuid4().hex
        pdf_name, jpg_name = save_result_pages(pages, stamp, next_job_id)
        return jsonify(
            ok=True,
            pdf="/output/" + quote(pdf_name),
            preview="/output/" + quote(jpg_name),
            job_id=next_job_id,
            preview_page_height=A4_H,
            preview_gap=24,
            pdf_pages=len(pages),
        )
    except Exception as exc:
        app.logger.exception("预览后手动擦除失败")
        return jsonify(ok=False, error=f"应用擦除标记失败：{exc}"), 500


if __name__ == "__main__":
    cfg = load_config()
    print("=" * 50)
    print("错题白卷生成器（OCR+楷体混合版）")
    if cfg["ok"]:
        print("密钥：已配置")
        print("模式：", cfg.get("mode", "hybrid"))
    else:
        print("密钥：未配置 →", cfg["reason"])
        print("请配置环境变量或 config.local.json 中的腾讯云密钥")
    print("启动后浏览器自动打开，地址 http://127.0.0.1:7860")
    print("=" * 50)
    import webbrowser
    webbrowser.open("http://127.0.0.1:7860")
    app.run(host="127.0.0.1", port=7860, debug=False)
