# -*- coding: utf-8 -*-
"""OCR + 楷体重排，以及带图题的版面拆分拼装。"""
from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# 题号行特征
_RE_QNUM = re.compile(r"^\s*\d{1,3}\s*[\.．、\)）]")
# 括号内容（用于清空答案）
_RE_BLANK = re.compile(r"([（(])([^（）()]{0,40}?)([）)])")
# 像句末碎片（常被 OCR 拆到前面）
_RE_TAIL_FRAG = re.compile(r"^[）)\s。．、，,；;：:秒分千克米]*(?:。)?$")
# 纯数字噪点行（手写批注「7 7 7」）
_RE_DIGIT_NOISE = re.compile(r"^[\d０-９\s]{1,12}$")
# LaTeX 分数 → 纯文本
_RE_LATEX_FRAC = re.compile(
    r"\$?\s*\\frac\s*\{\s*([^{}]*)\s*\}\s*\{\s*([^{}]*)\s*\}\s*\$?"
)
_RE_LATEX_DOLLAR = re.compile(r"\$+")


@dataclass
class OcrLine:
    text: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)

    @property
    def width(self) -> int:
        return max(1, self.x2 - self.x1)

    @property
    def area(self) -> int:
        return self.width * self.height


def find_kaiti_font(explicit: str = "") -> str:
    """定位 Windows 楷体字体文件。"""
    candidates = []
    if explicit:
        candidates.append(explicit)
    windir = os.environ.get("WINDIR", r"C:\Windows")
    fonts = os.path.join(windir, "Fonts")
    candidates.extend(
        [
            os.path.join(fonts, "simkai.ttf"),
            os.path.join(fonts, "SIMKAI.TTF"),
            os.path.join(fonts, "STKAITI.TTF"),
            os.path.join(fonts, "stkaiti.ttf"),
            os.path.join(fonts, "KaiTi.ttf"),
        ]
    )
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return candidates[1] if len(candidates) > 1 else "simkai.ttf"


def image_to_b64(img: Image.Image, fmt: str = "JPEG", quality: int = 95) -> str:
    buf = io.BytesIO()
    rgb = img.convert("RGB")
    if fmt.upper() == "PNG":
        rgb.save(buf, format="PNG", optimize=True)
    else:
        rgb.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _det_bbox(det) -> Optional[Tuple[int, int, int, int]]:
    """优先用原图 Polygon，避免 ItemPolygon（旋转纠正坐标系）导致行序错乱。"""
    if det.Polygon:
        xs = [int(p.X) for p in det.Polygon]
        ys = [int(p.Y) for p in det.Polygon]
        return min(xs), min(ys), max(xs), max(ys)
    ip = det.ItemPolygon
    if ip is not None and ip.Width and ip.Height:
        return int(ip.X), int(ip.Y), int(ip.X + ip.Width), int(ip.Y + ip.Height)
    return None


def reconstruct_reading_order(lines: List[OcrLine], page_w: int = 0) -> List[OcrLine]:
    """把碎块按「行带」聚类；同行仅在间距合理时从左到右合并。"""
    if not lines:
        return []
    if page_w <= 0:
        page_w = max(L.x2 for L in lines)
    ordered = sorted(lines, key=lambda L: (L.cy, L.x1))
    heights = sorted(L.height for L in ordered)
    med_h = heights[len(heights) // 2]
    thresh = max(12.0, med_h * 0.55)

    rows: List[List[OcrLine]] = []
    cur: List[OcrLine] = []
    row_cy = 0.0
    for L in ordered:
        if not cur:
            cur = [L]
            row_cy = L.cy
            continue
        if abs(L.cy - row_cy) <= thresh:
            cur.append(L)
            row_cy = sum(x.cy for x in cur) / len(cur)
        else:
            rows.append(cur)
            cur = [L]
            row_cy = L.cy
    if cur:
        rows.append(cur)

    def should_join(a: OcrLine, b: OcrLine) -> bool:
        """控制同行合并：避免把上下两道竖排小题粘成一行。"""
        gap = b.x1 - a.x2
        ta, tb = a.text.strip(), b.text.strip()
        if not ta or not tb:
            return True
        # 上一截以未闭合括号结尾，下一截像新句子 → 不合并（第3题典型）
        if ta.endswith(("（", "(")) and re.match(r"^[\u4e00-\u9fff]", tb):
            return False
        # 两段都是较长中文陈述 → 不合并
        han_a = sum(1 for ch in ta if "\u4e00" <= ch <= "\u9fff")
        han_b = sum(1 for ch in tb if "\u4e00" <= ch <= "\u9fff")
        if han_a >= 6 and han_b >= 4 and gap > max(24, page_w * 0.06):
            return False
        # 紧挨着 → 合并
        if gap < max(16, page_w * 0.035):
            return True
        # 换算题同行两项：…秒 / …时 后面接数字开头
        if re.search(r"(秒|时|米|毫米|厘米|分米|千克|本)\s*$", ta) and re.match(
            r"^\d", tb
        ):
            return True
        # 比大小同行：左侧结束、右侧数字
        if gap < page_w * 0.22 and re.match(r"^\d", tb):
            return True
        if gap > page_w * 0.14:
            return False
        return True

    merged: List[OcrLine] = []
    for row in rows:
        row = sorted(row, key=lambda L: L.x1)
        # 可能拆成多段（大间距或不应合并）
        groups: List[List[OcrLine]] = [[row[0]]]
        for L in row[1:]:
            if should_join(groups[-1][-1], L):
                groups[-1].append(L)
            else:
                groups.append([L])
        for group in groups:
            parts = []
            for i, L in enumerate(group):
                t = L.text
                if i > 0 and parts:
                    prev = parts[-1]
                    if prev and t:
                        # 仅「比大小」语境才插 ○；换算题（含 =）只用空格
                        if (
                            "=" not in prev
                            and "=" not in t
                            and re.search(r"(分米|(?<![千分厘])米)\s*$", prev)
                            and re.match(r"^\d", t)
                        ):
                            if "○" not in prev[-4:]:
                                parts.append(" ○ ")
                            else:
                                parts.append(" ")
                        elif (prev[-1].isalnum() or prev[-1] in "+-×÷=<>）)秒时本") and (
                            t[0].isalnum() or t[0] in "+-×÷=<>.（("
                        ):
                            parts.append("  ")
                parts.append(t)
            text = "".join(parts)
            merged.append(
                OcrLine(
                    text=text,
                    confidence=sum(L.confidence for L in group) / len(group),
                    x1=min(L.x1 for L in group),
                    y1=min(L.y1 for L in group),
                    x2=max(L.x2 for L in group),
                    y2=max(L.y2 for L in group),
                )
            )

    if len(merged) >= 2:
        first, last = merged[0].text.strip(), merged[-1].text.strip()
        if (not _RE_QNUM.match(first)) and _RE_QNUM.match(last):
            if _RE_TAIL_FRAG.match(first) or (
                len(first) <= 6 and first[:1] in "）)。．秒分"
            ):
                merged = list(reversed(merged))
        elif _RE_QNUM.match(last) and not _RE_QNUM.match(first) and first.startswith(("）", ")")):
            merged = list(reversed(merged))
    return merged


def ocr_lines(client, img: Image.Image, jpeg_quality: int = 95) -> List[OcrLine]:
    """调用 GeneralAccurateOCR，返回按阅读顺序合并后的文本行。"""
    from tencentcloud.ocr.v20181119 import models

    req = models.GeneralAccurateOCRRequest()
    req.ImageBase64 = image_to_b64(img, "JPEG", jpeg_quality)
    req.EnableDetectSplit = False
    try:
        req.WordsType = "2"
    except Exception:
        pass

    resp = client.GeneralAccurateOCR(req)
    raw: List[OcrLine] = []
    for det in resp.TextDetections or []:
        text = (det.DetectedText or "").strip()
        if not text:
            continue
        box = _det_bbox(det)
        if not box:
            continue
        x1, y1, x2, y2 = box
        raw.append(
            OcrLine(
                text=text,
                confidence=float(det.Confidence or 0),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    raw = filter_raw_detections(raw)
    return reconstruct_reading_order(raw, page_w=img.size[0])


def assess_erased_question(
    client, img: Image.Image, jpeg_quality: int = 95
) -> List[str]:
    """用 OCR 只做质量检查，不用识别结果改写题目。"""
    try:
        lines = ocr_lines(client, img, jpeg_quality)
    except Exception as exc:
        return [f"自动质量检查失败：{exc}"]
    texts = [normalize_math_text(line.text) for line in lines]
    warnings: List[str] = []
    if audit_question_texts(texts):
        warnings.append("检测到括号、单位或算式可能不完整")
    joined = "\n".join(texts)
    if re.search(r"=\s*[-+]?\d+(?:\.\d+)?(?:\s|$|[）)])", joined):
        warnings.append("等号后仍识别到数字，请检查是否有答案残留")
    if re.search(r"[（(]\s*[-+]?\d+(?:\.\d+)?(?:\s*[^（）()]{0,6})?[）)]", joined):
        warnings.append("括号内仍识别到数字，请检查是否有答案残留")
    return list(dict.fromkeys(warnings))


def is_noise_fragment(text: str, conf: float, height: int, med_h: float) -> bool:
    """判断是否为手写批注/碎字，不应进入白卷。"""
    t = (text or "").strip()
    if not t:
        return True
    if _RE_QNUM.match(t):
        return False
    # 数字可能是竖式、数列或配图条件，不能仅凭内容判定为噪声。
    if _RE_DIGIT_NOISE.fullmatch(t):
        return False
    # 明显偏矮的小字（分数旁手写标注）
    if med_h > 0 and height < med_h * 0.45 and len(t) <= 6:
        return True
    # 低置信且几乎无汉字
    han = sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff")
    if conf < 70 and han == 0 and len(t) <= 8:
        return True
    return False


def normalize_math_text(text: str) -> str:
    """去掉 LaTeX 分数写法，改成 2/7 这种可印刷形式。"""
    if not text:
        return text
    # 先处理转义符号，避免 \% 被吃成残片
    text = (
        text.replace("\\times", "×")
        .replace("\\div", "÷")
        .replace("\\circ", "○")
        .replace("\\bigcirc", "○")
        .replace("\\%", "%")
        .replace("\\leqslant", "≤")
        .replace("\\geqslant", "≥")
    )
    text = _RE_LATEX_FRAC.sub(
        lambda m: (
            f"{(m.group(1) or '').strip() or '（　　）'}"
            f"/{(m.group(2) or '').strip() or '（　　）'}"
        ),
        text,
    )
    text = _RE_LATEX_DOLLAR.sub("", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("{", "").replace("}", "")
    # 空分子/分母
    text = re.sub(r"\(\s*\)\s*/", "（　　）/", text)
    text = re.sub(r"/\s*\(\s*\)", "/（　　）", text)
    # 常见误识别
    text = re.sub(r"(?<![千])个米", "千米", text)
    text = text.replace("跑了了", "跑了")
    # 进位手写「1」被读成「1/259+…」（不做跨试卷猜题）
    text = re.sub(r"(?<![0-9])1\s*/\s*(?=\d{2,4}\s*[＋+])", "", text)
    text = text.replace("%_0", "%").replace("%_", "%")
    text = re.sub(r"%\s*[／/]\s*0\b", "%", text)
    return text.strip()


def looks_like_elementary_compare(joined: str) -> bool:
    """仅三年级风格「比大小 / 分米对照」，避免六年级填○题被误重建。"""
    if not joined:
        return False
    if "比大小" in joined:
        return True
    if "分米" in joined and re.search(r"\d+\s*米", joined):
        return True
    return False


def looks_like_circle_fill_compare(joined: str) -> bool:
    """六年级「在○内填上 > < =」类。"""
    return bool(
        re.search(r"在\s*[○O〇0]\s*内", joined)
        or re.search(r"填上\s*[\"“]?[>＜<]=", joined)
        or (">" in joined and "<" in joined and "=" in joined and "填" in joined)
    )


def is_broken_compare_ocr(texts: List[str]) -> bool:
    """比大小残片：缺 ○/右侧。"""
    joined = " ".join(normalize_math_text(t or "") for t in texts if t)
    if not looks_like_elementary_compare(joined) and "○" not in joined:
        if not re.search(r"\d+\s*[＋+]\s*\d+", joined):
            return False
    if re.search(r"\d+\s*[＋+×xX*＊]\s*\d+", joined) and "○" not in joined:
        if not re.search(
            r"[＋+×xX*＊]\s*\d+\D{0,10}?(?<![0-9])(\d{3,5})(?![0-9/])", joined
        ):
            return True
    if re.fullmatch(r"[\d＋+\-×xX*＊/\s]{3,24}", joined) and "○" not in joined:
        return True
    return False


def is_poor_circle_fill_ocr(texts: List[str]) -> bool:
    """填○比较题 OCR 严重失真（几乎没有小数/完整算式）时应用擦除。"""
    joined = " ".join(normalize_math_text(t or "") for t in texts if t)
    if not looks_like_circle_fill_compare(joined):
        return False
    decimals = len(re.findall(r"\d+\.\d+", joined))
    circles = joined.count("○") + joined.count("O")
    # 正常六小题应有多个小数；若几乎没有，说明 OCR 崩了
    if decimals < 2 and circles < 3:
        return True
    if re.search(r"98\s*[×xX]\s*0\b", joined) and "6.98" not in joined:
        return True
    return False


def dedupe_ocr_texts(texts: List[str]) -> List[str]:
    """去掉 OCR 重复行 / 同一长句粘贴多次 / 整段内容复制两遍。"""
    from difflib import SequenceMatcher

    def norm_key(s: str) -> str:
        s = re.sub(r"[\s　（）()\[\]【】]+", "", s or "")
        s = re.sub(r"[：:．.。，,、]+", "", s)
        return s

    def similar(a: str, b: str) -> float:
        ka, kb = norm_key(a), norm_key(b)
        if not ka or not kb:
            return 0.0
        if ka == kb:
            return 1.0
        if len(ka) >= 12 and (ka in kb or kb in ka):
            return 0.96
        return SequenceMatcher(None, ka, kb).ratio()

    raw = [(t or "").strip() for t in texts if (t or "").strip()]
    # 单行内部：整句复制两遍
    fixed = []
    for t in raw:
        half = len(t) // 2
        if half > 16:
            left, right = t[:half].strip(), t[half:].strip()
            if similar(left, right) >= 0.9:
                t = left if len(left) >= len(right) else right
        # 三遍重复（少见）
        third = len(t) // 3
        if third > 20:
            a, b, c = t[:third], t[third : 2 * third], t[2 * third :]
            if similar(a, b) >= 0.9 and similar(b, c) >= 0.9:
                t = a.strip()
        fixed.append(t)

    out: List[str] = []
    for t in fixed:
        if out and similar(out[-1], t) >= 0.9:
            # 保留更长、更完整的一行
            if len(norm_key(t)) > len(norm_key(out[-1])):
                out[-1] = t
            continue
        # 与更早几行高度重复也丢
        if any(similar(prev, t) >= 0.92 for prev in out[-4:]):
            continue
        out.append(t)

    # 整段：前半与后半是同一批题
    if len(out) >= 4 and len(out) % 2 == 0:
        mid = len(out) // 2
        a = "".join(out[:mid])
        b = "".join(out[mid:])
        if similar(a, b) >= 0.88:
            out = out[:mid]
    elif len(out) >= 3:
        # 允许奇数：后半比前半少一行时，比最长公共前缀
        mid = len(out) // 2
        if mid >= 2:
            a = "".join(out[:mid])
            b = "".join(out[mid : mid + mid])
            if b and similar(a, b) >= 0.88:
                out = out[:mid]
    return out


def looks_like_scale_or_diagram(joined: str) -> bool:
    """比例尺 / 示意图：文字 OCR 易崩，应整题擦除保留版面。"""
    if not joined:
        return False
    if re.search(r"比例尺|图上距离|实际距离|线段图|见下图|如下图", joined):
        return True
    # OCR 把比例尺读成 0/L20km、1/L 等残片
    if re.search(r"0\s*/\s*L|/\s*L\s*\d|\d\s*/\s*L\s*\d", joined, re.I):
        return True
    if re.search(r"比例|图上|实际", joined) and re.search(
        r"(km|千米|米|厘米)", joined
    ):
        return True
    return False


# 常见计量单位（填空括号后）
_UNITS = (
    "千米|毫米|厘米|分米|平方米|立方米|公顷|千克|克|吨|"
    "小时|分钟|秒钟|秒|时|分|米|本|个|元|角|分"
)


def blank_paren_answers(text: str) -> str:
    """把括号内的短答案清空，便于重练。"""

    # OCR 文本不包含可靠的手写来源信息；仅统一本来就为空的括号。
    return re.sub(r"[（(]\s*[）)]", "（　　）", text)


def repair_blank_phrases(text: str) -> str:
    """修复 OCR 把填空括号读残的情况。"""
    if not text:
        return text
    u = _UNITS

    text = re.sub(r"分\s*[（(]\s*秒", "分（　　）秒", text)
    text = re.sub(
        r"需要\s*[（(]\s*[）)]?\s*分\s*[（(]\s*秒",
        "需要（　　）分（　　）秒",
        text,
    )
    text = re.sub(r"共占\s*[（(]\s*[）)]?", "共占（　　）", text)
    text = re.sub(r"^[）)]\s*分\s*[（(]", "（　　）分（　　）", text)

    text = re.sub(
        rf"=\s*[（(]\s*[0-9０-９./／⁄]{{0,8}}\s*({u})",
        r"=（　　）\1",
        text,
    )
    text = re.sub(rf"=\s*[（(]\s*({u})", r"=（　　）\1", text)

    # 选择单位：为2(儿童… / 长15( / 行末 16(
    text = re.sub(
        r"(为|长|了|约|是|有)\s*([0-9０-９]+)\s*[（(]\s*(?=[\u4e00-\u9fff]|$)",
        r"\1\2（　　）",
        text,
    )
    text = re.sub(r"([0-9０-９])\s*[（(]\s*$", r"\1（　　）", text)
    text = re.sub(
        r"([0-9０-９])\s*[（(]\s*([\u4e00-\u9fff])",
        r"\1（　　）\2",
        text,
    )
    # 仅行末「大约为 2 / 长 15」缺括号时补全（避免误伤「边长8厘米」「行驶了15千米」）
    text = re.sub(
        r"(为|长|了|约)\s*([0-9０-９]+)\s*$",
        r"\1\2（　　）",
        text,
    )

    text = re.sub(
        rf"(卖出了|一共|共|是|为|得|等于)\s*[（(]\s*[0-9０-９./／⁄]{{0,8}}\s*({u})",
        r"\1（　　）\2",
        text,
    )
    text = re.sub(
        rf"(卖出了|一共|共|是|为|得|等于)\s*[（(]\s*({u})",
        r"\1（　　）\2",
        text,
    )

    text = re.sub(r"[（(]\s*[）)]", "（　　）", text)
    return text


def repair_compare_phrases(text: str) -> str:
    """修复「比大小」圆圈与算式粘连。"""
    if not text:
        return text
    text = text.replace("◯", "○").replace("〇", "○").replace("∘", "○")
    text = re.sub(r"(\d+\s*分米)\s*[Oo×xX＊*]?\s*(\d+\s*米\b)", r"\1 ○ \2", text)
    text = re.sub(r"(\d+\s*米)\s*[Oo×xX＊*]?\s*(\d+\s*分米)", r"\1 ○ \2", text)
    text = re.sub(
        r"(\d+\s*[＋+]\s*\d+)\s*[Oo×xX＊*]?\s*(\d{3,5})(?!\d)",
        r"\1 ○ \2",
        text,
    )
    text = re.sub(
        r"(\d+\s*[×xX*＊]\s*\d+)\s*[Oo×xX＊*]?\s*(\d{3,5})(?!\d)",
        r"\1 ○ \2",
        text,
    )
    text = re.sub(
        r"(\d+\s*/\s*\d+)\s*[Oo×xX＊*]?\s*(\d+\s*/\s*\d+)",
        r"\1 ○ \2",
        text,
    )
    if "比大小" in text:
        text = re.sub(r"○{2,}", "○", text)
    # 换算题两项之间误插的 ○（如：…千米 ○ 3厘米=…）
    text = re.sub(
        r"(=(?:（　　）)?(?:千米|毫米|厘米|分米|时|秒|本))\s*○\s*(?=\d)",
        r"\1  ",
        text,
    )
    return text


def expand_glued_lines(text: str) -> List[str]:
    """把误粘在一起的小题拆开：…（　　）儿童口罩…"""
    if not text:
        return []
    # 填空后紧跟新的中文小题开头
    text = re.sub(
        r"（　　）(?=[\u4e00-\u9fff]{2,})",
        "（　　）\n",
        text,
    )
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    return parts or [text]


def filter_raw_detections(lines: List[OcrLine]) -> List[OcrLine]:
    """合并前先丢掉批注碎块，避免合成「7 7 7」。"""
    if not lines:
        return []
    heights = sorted(L.height for L in lines)
    med_h = float(heights[len(heights) // 2])
    kept = []
    for L in lines:
        if is_noise_fragment(L.text, L.confidence, L.height, med_h):
            continue
        kept.append(L)
    return kept


def clean_ocr_lines(lines: List[OcrLine], min_conf: float = 55.0) -> List[OcrLine]:
    """过滤低置信度纯手写行，规范化数学/填空。"""
    out: List[OcrLine] = []
    heights = sorted(L.height for L in lines) if lines else [20]
    med_h = float(heights[len(heights) // 2])
    for L in lines:
        text = normalize_math_text(L.text)
        text = blank_paren_answers(text)
        text = repair_blank_phrases(text)
        text = repair_compare_phrases(text)
        text = text.strip()
        if not text:
            continue
        pieces = expand_glued_lines(text)
        for pi, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            if is_noise_fragment(piece, L.confidence, L.height, med_h):
                continue
            is_q = bool(_RE_QNUM.match(piece))
            if L.confidence < min_conf and not is_q:
                if len(piece) <= 12 and not any(
                    ch in piece for ch in "（）()？?．。，,：:/○"
                ):
                    continue
            # 拆行后略微错开 y，保持顺序
            y_off = pi * max(2, L.height // 3)
            out.append(
                OcrLine(
                    text=piece,
                    confidence=L.confidence,
                    x1=L.x1,
                    y1=L.y1 + y_off,
                    x2=L.x2,
                    y2=L.y2 + y_off,
                )
            )
    return out


def text_coverage(lines: List[OcrLine], w: int, h: int) -> float:
    if w <= 0 or h <= 0 or not lines:
        return 0.0
    area = sum(L.area for L in lines)
    return min(1.0, area / float(w * h))


def max_vertical_gap_ratio(lines: List[OcrLine], h: int) -> Tuple[float, Optional[Tuple[int, int]]]:
    """返回最大无文字垂直空隙占比，以及空隙 (y0,y1)。"""
    if h <= 0 or not lines:
        return 1.0, (0, h)
    intervals = sorted((L.y1, L.y2) for L in lines)
    merged = []
    for a, b in intervals:
        if not merged or a > merged[-1][1] + 2:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    gaps = []
    prev = 0
    for a, b in merged:
        if a - prev > 0:
            gaps.append((prev, a))
        prev = b
    if prev < h:
        gaps.append((prev, h))
    if not gaps:
        return 0.0, None
    best = max(gaps, key=lambda g: g[1] - g[0])
    return (best[1] - best[0]) / float(h), best


def region_has_ink(img: Image.Image, bbox: Tuple[int, int, int, int], dark_ratio: float = 0.02) -> bool:
    """配图候选区是否有足够深色像素（避免把题下空白当图）。"""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return False
    crop = img.crop((x1, y1, x2, y2)).convert("L")
    crop.thumbnail((200, 200))
    hist = crop.histogram()
    dark = sum(hist[:180])
    total = max(1, crop.size[0] * crop.size[1])
    return (dark / total) >= dark_ratio


def _overlap_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def find_side_figure_bbox(
    lines: List[OcrLine],
    w: int,
    h: int,
    src_img: Image.Image,
    pad: int = 6,
    band_y1: Optional[int] = None,
    band_y2: Optional[int] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """检测左右侧配图（如「如右图」）：右/左半边少字且有墨迹。
    可限定在 band_y1~band_y2 竖直带内（整页框选时只在「如图」题附近找）。"""
    if w < 80 or h < 80 or src_img is None:
        return None
    y1 = int(band_y1) if band_y1 is not None else 0
    y2 = int(band_y2) if band_y2 is not None else h
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 40, min(y2, h))
    candidates = []
    # 右侧 40% / 左侧 40%
    for x1, x2 in ((int(w * 0.52), w - pad), (pad, int(w * 0.48))):
        if x2 - x1 < w * 0.25:
            continue
        box = (x1, y1 + pad, x2, y2 - pad)
        if box[3] <= box[1] or box[2] <= box[0]:
            continue
        text_area = 0
        for L in lines:
            text_area += _overlap_area(box, (L.x1, L.y1, L.x2, L.y2))
        box_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        if text_area / box_area > 0.28:
            continue
        if region_has_ink(src_img, box, dark_ratio=0.02):
            candidates.append((box_area - text_area, box))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


_RE_FIGURE_CUE = re.compile(r"如[左右上下]?图|看图|见图|下图|右图")


def find_figure_by_cue(
    lines: List[OcrLine],
    w: int,
    h: int,
    src_img: Image.Image,
    pad: int = 6,
) -> Optional[Tuple[int, int, int, int]]:
    """根据「如图/如右图」题干，在该题竖直带内找侧配图。"""
    if not lines or src_img is None:
        return None
    cues = [L for L in lines if _RE_FIGURE_CUE.search(L.text or "")]
    if not cues:
        return None
    # 取第一个线索；带高扩展到下一题号或固定高度
    cue = min(cues, key=lambda L: L.y1)
    qnums = sorted(
        (L.y1 for L in lines if _RE_QNUM.match((L.text or "").strip())),
    )
    band_top = max(0, cue.y1 - pad)
    band_bot = min(h, cue.y2 + max(120, int(h * 0.22)))
    for qy in qnums:
        if qy > cue.y2 + 8:
            band_bot = min(band_bot, qy - pad)
            break
    # 同题其它文字行也纳入带高
    for L in lines:
        if L.y1 >= band_top and L.y2 <= band_bot + 40 and L.cy < band_bot:
            band_bot = max(band_bot, min(h, L.y2 + pad))
    fig = find_side_figure_bbox(
        lines, w, h, src_img, pad=pad, band_y1=band_top, band_y2=band_bot
    )
    if fig is None:
        # 再放宽：线索行右侧矩形
        x1 = min(w - 40, max(int(w * 0.45), cue.x2 + pad))
        box = (x1, band_top, w - pad, band_bot)
        if region_has_ink(src_img, box, dark_ratio=0.015):
            fig = box
    return fig


def find_figure_bbox(
    lines: List[OcrLine],
    w: int,
    h: int,
    pad: int = 8,
    min_w_ratio: float = 0.25,
    min_h_ratio: float = 0.18,
    min_area_ratio: float = 0.08,
    src_img: Optional[Image.Image] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """配图区域：先按「如图」线索，再试上下空隙，再试整页左右侧图。"""
    if src_img is not None:
        by_cue = find_figure_by_cue(lines, w, h, src_img, pad=pad)
        if by_cue is not None:
            return by_cue

    gap_ratio, gap = max_vertical_gap_ratio(lines, h)
    vert = None
    if gap is not None and gap_ratio >= min_h_ratio:
        gy0, gy1 = gap
        x1, x2 = pad, w - pad
        for L in lines:
            overlap = max(0, min(L.y2, gy1) - max(L.y1, gy0))
            if overlap > 0.5 * max(1, L.y2 - L.y1):
                mid = (L.x1 + L.x2) / 2
                if mid < w * 0.35:
                    x1 = max(x1, L.x2 + pad)
                elif mid > w * 0.65:
                    x2 = min(x2, L.x1 - pad)
        if x2 - x1 < w * min_w_ratio:
            x1, x2 = pad, w - pad
        y1 = max(0, gy0 + pad // 2)
        y2 = min(h, gy1 - pad // 2)
        if (
            y2 - y1 >= h * min_h_ratio
            and (x2 - x1) * (y2 - y1) >= w * h * min_area_ratio
        ):
            bbox = (int(x1), int(y1), int(x2), int(y2))
            if src_img is None or region_has_ink(src_img, bbox):
                vert = bbox

    side = None
    if src_img is not None:
        side = find_side_figure_bbox(lines, w, h, src_img, pad=pad)

    if vert and side:
        va = (vert[2] - vert[0]) * (vert[3] - vert[1])
        sa = (side[2] - side[0]) * (side[3] - side[1])
        return vert if va >= sa else side
    return vert or side


def lines_outside_figure(
    lines: List[OcrLine], figure: Optional[Tuple[int, int, int, int]]
) -> List[OcrLine]:
    """去掉落在配图内的 OCR（如图上「30千克」），避免排进题干。"""
    if not figure:
        return lines
    out = []
    for L in lines:
        box = (L.x1, L.y1, L.x2, L.y2)
        inter = _overlap_area(box, figure)
        if inter >= 0.45 * L.area:
            continue
        out.append(L)
    return out


def classify_layout(
    lines: List[OcrLine],
    w: int,
    h: int,
    cfg: dict,
    src_img: Optional[Image.Image] = None,
) -> str:
    """返回 text_only | mixed | fallback。"""
    if not lines:
        return "fallback"
    avg_conf = sum(L.confidence for L in lines) / len(lines)
    min_conf = float(cfg.get("ocr_min_avg_conf", 60))
    min_lines = int(cfg.get("ocr_min_lines", 1))
    if len(lines) < min_lines or avg_conf < min_conf:
        return "fallback"

    joined = "".join(L.text for L in lines)
    # 「如图」类题优先走图文混合
    if src_img is not None and _RE_FIGURE_CUE.search(joined):
        fig_cue = find_figure_by_cue(lines, w, h, src_img)
        if fig_cue is not None:
            return "mixed"

    cov = text_coverage(lines, w, h)
    gap_ratio, _ = max_vertical_gap_ratio(lines, h)
    cov_high = float(cfg.get("text_coverage_high", 0.12))
    gap_mixed = float(cfg.get("figure_gap_ratio", 0.18))

    fig = find_figure_bbox(
        lines,
        w,
        h,
        min_h_ratio=float(cfg.get("figure_min_h_ratio", 0.18)),
        min_w_ratio=float(cfg.get("figure_min_w_ratio", 0.25)),
        min_area_ratio=float(cfg.get("figure_min_area_ratio", 0.08)),
        src_img=src_img,
    )
    if fig is not None and (
        gap_ratio >= gap_mixed
        or fig[0] > w * 0.4
        or fig[2] < w * 0.6
        or _RE_FIGURE_CUE.search(joined)
    ):
        return "mixed"
    if cov >= cov_high or gap_ratio < gap_mixed or fig is None:
        return "text_only"
    if cov < float(cfg.get("text_coverage_low", 0.04)):
        return "fallback"
    return "text_only"


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> List[str]:
    """按像素宽度换行；禁止在填空括号内断开，优先在标点/单位后断行。"""
    if not text:
        return [""]

    protected: List[str] = []

    def _protect(m):
        protected.append(m.group(0))
        return f"\x01{len(protected)-1}\x02"

    def _restore(s: str) -> str:
        for i, p in enumerate(protected):
            s = s.replace(f"\x01{i}\x02", p)
        return s

    def _width(s: str) -> int:
        bbox = draw.textbbox((0, 0), _restore(s), font=font)
        return bbox[2] - bbox[0]

    # 整段填空视为原子（全角空格填空）
    work = re.sub(r"[（(][　\s]{0,10}[）)]", _protect, text)
    # 也能保护未闭合很少见的情况：不在此拆

    if _width(work) <= max_w:
        return [_restore(work)]

    prefer_break = set("，。；、：:）)○　 ")
    lines: List[str] = []
    buf = ""
    i = 0
    while i < len(work):
        # 保护占位符整体吃进
        if work[i] == "\x01":
            j = work.find("\x02", i)
            if j < 0:
                j = i
            chunk = work[i : j + 1]
            if buf and _width(buf + chunk) > max_w:
                lines.append(_restore(buf))
                buf = chunk
            else:
                buf += chunk
            i = j + 1
            continue
        ch = work[i]
        trial = buf + ch
        if _width(trial) <= max_w:
            buf = trial
            i += 1
            continue
        # 需要断行：向前找合适断点；避免在「=」或未闭合括号后硬断
        break_at = -1
        for k in range(len(buf) - 1, max(-1, len(buf) - 28), -1):
            if buf[k] in prefer_break:
                break_at = k + 1
                break
            if k >= 1 and buf[k - 1 : k + 1] in ("千米", "毫米", "厘米", "分米", "千克"):
                break_at = k + 1
                break
            if k >= 0 and buf[k] in "秒时本岁倍吨":
                break_at = k + 1
                break
        # 若断点落在未闭合的「=（」附近，尽量整段换行
        if break_at > 0:
            head = buf[:break_at]
            if head.rstrip().endswith(("=", "（", "(")) or (
                head.count("（") + head.count("(") > head.count("）") + head.count(")")
            ):
                break_at = -1
        if break_at <= 0:
            if buf:
                lines.append(_restore(buf))
            buf = ch
        else:
            lines.append(_restore(buf[:break_at]))
            buf = buf[break_at:] + ch
        i += 1
    if buf:
        lines.append(_restore(buf))
    return [ln for ln in lines if ln is not None] or [""]


def coalesce_continuations(texts: List[str]) -> List[str]:
    """把被错误拆到下一行的单位/括号碎片拼回上一行。"""
    if not texts:
        return texts
    out: List[str] = []
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue
        if not out:
            out.append(t)
            continue
        prev = out[-1]
        # 下一行只有单位或「秒。」「千米。」「厘米。」
        if re.fullmatch(
            r"[）)]?[　\s]*(秒|时|分|米|本|岁|千米|毫米|厘米|分米|千克)[。．]?",
            t,
        ):
            out[-1] = prev.rstrip() + t.lstrip()
            continue
        # 上一行以未闭合括号或「分（」「长（」结尾
        if re.search(r"[（(]\s*$", prev) or re.search(r"[（(][　\s]*$", prev):
            out[-1] = prev.rstrip() + t.lstrip()
            continue
        # 「跑了」/「长」断行：上一行以「了（」或「了」结尾，下一行「了（　　）」或「（　　）千米」
        if prev.endswith(("了", "了（", "了(")) and re.match(
            r"^(了?\s*[（(]|[（(]|千米|秒)", t
        ):
            # 去掉重复的「了」
            if prev.endswith("了") and t.startswith("了"):
                t = t[1:].lstrip()
            out[-1] = prev.rstrip() + t.lstrip()
            continue
        # 上一行以 =（ 或数字（ 结尾且未闭合
        if re.search(r"[=为约长了]\s*[（(]\s*$", prev) or (
            prev.count("（") + prev.count("(") > prev.count("）") + prev.count(")")
            and not re.search(r"[。．？?]$", prev)
            and len(t) <= 12
        ):
            if not _RE_QNUM.match(t):
                out[-1] = prev.rstrip() + t.lstrip()
                continue
        out.append(t)
    # 再跑一轮短语修复
    return [repair_blank_phrases(repair_compare_phrases(x)) for x in out]


def merge_broken_blank_lines(texts: List[str]) -> List[str]:
    """合并因 OCR/换行导致的括号断裂。"""
    if not texts:
        return texts
    out = []
    i = 0
    while i < len(texts):
        cur = texts[i]
        while i + 1 < len(texts):
            nxt = texts[i + 1]
            if cur.rstrip().endswith(("（", "(")) or re.search(
                r"[（(][　\s]*$", cur.rstrip()
            ):
                cur = cur.rstrip() + nxt.lstrip()
                i += 1
                continue
            if re.match(r"^[）)\s　]*(秒|时|分|米|本|岁|千米|毫米|厘米|分米|千克)", nxt):
                cur = cur.rstrip() + nxt.lstrip()
                i += 1
                continue
            if re.match(r"^[）)]\s*分", nxt) and cur:
                cur = cur.rstrip() + nxt.lstrip()
                i += 1
                continue
            break
        out.append(repair_blank_phrases(cur))
        i += 1
    return coalesce_continuations(out)


@dataclass
class TextIssue:
    code: str
    detail: str
    line_idx: int = -1


_RE_ORPHAN_UNIT = re.compile(
    r"^[）)]?[　\s]*(秒|时|分|米|本|岁|千米|毫米|厘米|分米|千克)[。．]?$"
)
_RE_FALSE_CIRCLE_CONV = re.compile(
    r"(=(?:（　　）)?(?:千米|毫米|厘米|分米|时|秒|本))\s*○\s*(?=\d)"
)


def audit_question_texts(texts: List[str]) -> List[TextIssue]:
    """生成后校验：找出已知会破坏卷面的排版/OCR 残留问题。"""
    issues: List[TextIssue] = []
    for i, raw in enumerate(texts):
        t = (raw or "").strip()
        if not t:
            continue
        if _RE_ORPHAN_UNIT.match(t):
            issues.append(TextIssue("orphan_unit", f"单位碎片独占一行：{t}", i))
        open_c = t.count("（") + t.count("(")
        close_c = t.count("）") + t.count(")")
        if open_c > close_c:
            issues.append(TextIssue("unclosed_blank", f"括号未闭合：{t[:40]}", i))
        if t.rstrip().endswith(("=", "=（", "=(")) or re.search(
            r"[（(]\s*$", t
        ):
            issues.append(TextIssue("broken_blank_tail", f"填空断在行尾：{t[-20:]}", i))
        if _RE_FALSE_CIRCLE_CONV.search(t) or (
            t.count("=") >= 2 and "○" in t and "比大小" not in t
        ):
            issues.append(TextIssue("false_circle", f"换算题误插○：{t[:50]}", i))
        if "个米" in t:
            issues.append(TextIssue("ocr_typo", "「个米」应为「千米」", i))
        if "跑了了" in t:
            issues.append(TextIssue("ocr_typo", "「跑了了」重复", i))
        # 比大小右侧缺失：以乘法/加减结束且无 ○ 右侧
        if re.search(r"\d+\s*[×xX*＊＋+]\s*\d+\s*$", t) and "○" not in t[-8:]:
            if "比" in t or "大小" in "".join(texts[max(0, i - 1) : i + 1]):
                issues.append(
                    TextIssue("incomplete_compare", f"比大小右侧可能缺失：{t}", i)
                )
    # 跨行：上一行未闭合、下一行是续写
    for i in range(len(texts) - 1):
        a = (texts[i] or "").strip()
        b = (texts[i + 1] or "").strip()
        if not a or not b or _RE_QNUM.match(b):
            continue
        if a.count("（") + a.count("(") > a.count("）") + a.count(")"):
            issues.append(
                TextIssue("split_across_lines", f"括号跨行：…{a[-12:]} | {b[:12]}…", i)
            )
        if _RE_ORPHAN_UNIT.match(b):
            issues.append(TextIssue("orphan_unit_next", f"下一行单位碎片：{b}", i + 1))
    return issues


def _apply_proofread_fixes(texts: List[str]) -> List[str]:
    """针对校验出的问题做一轮机械修正。"""
    out: List[str] = []
    for t in texts:
        t = normalize_math_text(t or "")
        t = repair_blank_phrases(t)
        t = repair_compare_phrases(t)
        # 换算题里误插的 ○ 一律去掉（两项都含 =）
        if t.count("=") >= 2 and "○" in t and "比大小" not in t:
            t = re.sub(r"\s*○\s*", "  ", t)
        else:
            t = _RE_FALSE_CIRCLE_CONV.sub(r"\1  ", t)
        if t.strip():
            out.append(t.strip())
    out = merge_broken_blank_lines(out)
    out = coalesce_continuations(out)
    return out


def proofread_question_texts(
    texts: List[str], max_rounds: int = 3
) -> Tuple[List[str], List[TextIssue]]:
    """
    生成后自检闭环：检测已知错误 → 自动改正 → 再检，直到干净或达上限。
    返回 (修正后文本, 仍未消除的问题)。
    """
    cur = [t for t in texts if (t or "").strip()]
    last_issues: List[TextIssue] = []
    for _ in range(max_rounds):
        cur = _apply_proofread_fixes(cur)
        last_issues = audit_question_texts(cur)
        # 无法自动补全的 OCR 缺字（如比大小右侧数字）不阻塞其它修正
        blocking = [
            x
            for x in last_issues
            if x.code not in ("incomplete_compare",)
        ]
        if not blocking:
            return cur, last_issues
    return cur, last_issues


def proofread_wrapped_lines(lines: List[str]) -> List[str]:
    """换行后再校一次，避免把「秒。」「千米。」折到下一行。"""
    if not lines:
        return lines
    # 保留草稿空行：按非空段分别 coalesce
    out: List[str] = []
    buf: List[str] = []
    for ln in lines:
        if ln == "":
            if buf:
                out.extend(coalesce_continuations(buf))
                buf = []
            out.append("")
        else:
            buf.append(ln)
    if buf:
        out.extend(coalesce_continuations(buf))
    # 去掉 coalesce 可能产生的首尾多余空，但保留题间草稿空行
    return out


def render_question_text(
    texts: List[str],
    width: int,
    font_path: str,
    font_size: int = 42,
    line_gap: int = 18,
    padding: int = 8,
    draft_lines: int = 3,
) -> Image.Image:
    """把多行题干用楷体渲染成白底图，宽度固定为栏宽。
    相邻题号之间插入 draft_lines 行空白，供写草稿。"""
    font = _load_font(font_path, font_size)
    probe = Image.new("RGB", (width, 10), "white")
    draw = ImageDraw.Draw(probe)
    max_w = width - padding * 2
    wrapped: List[str] = []
    # 渲染前：自检 + 自动改正闭环
    texts, remain = proofread_question_texts(list(texts))
    texts = dedupe_ocr_texts(texts)
    if remain:
        print(
            "[proofread] 仍有问题：",
            "; ".join(f"{x.code}:{x.detail}" for x in remain[:8]),
        )
    prev_was_content = False
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue
        # 新题号前留草稿空行
        if prev_was_content and _RE_QNUM.match(t) and draft_lines > 0:
            for _ in range(draft_lines):
                wrapped.append("")
        lines = wrap_text(draw, t, font, max_w)
        wrapped.extend(lines)
        prev_was_content = True
    # 换行后再校：粘回被折断的单位/括号
    wrapped = proofread_wrapped_lines(wrapped)
    if not wrapped:
        wrapped = [""]

    line_h = font_size + line_gap
    height = padding * 2 + line_h * len(wrapped)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = padding
    for line in wrapped:
        if line:
            draw.text((padding, y), line, font=font, fill=(20, 20, 20))
        y += line_h
    bbox = img.getbbox()
    if bbox:
        # 保留底部草稿空白：不要裁掉末尾空行
        bottom = max(bbox[3] + padding, height)
        img = img.crop((0, 0, width, min(height, bottom)))
    return img


def assemble_blocks(blocks: List[Image.Image], width: int, gap: int = 24) -> Image.Image:
    """竖向拼接若干块，统一到 width。"""
    if not blocks:
        return Image.new("RGB", (width, 40), "white")
    scaled = []
    for b in blocks:
        if b.width != width:
            nh = max(1, int(b.height * (width / b.width)))
            b = b.resize((width, nh), Image.LANCZOS)
        scaled.append(b)
    total_h = sum(b.height for b in scaled) + gap * (len(scaled) - 1)
    canvas = Image.new("RGB", (width, total_h), "white")
    y = 0
    for i, b in enumerate(scaled):
        canvas.paste(b, (0, y))
        y += b.height + (gap if i < len(scaled) - 1 else 0)
    return canvas


def split_lines_by_figure(
    lines: List[OcrLine], figure: Tuple[int, int, int, int]
) -> Tuple[List[OcrLine], List[OcrLine]]:
    """按配图位置拆成图前/图后文本。侧图时全部文本视为图旁（放图前）。"""
    fx1, fy1, fx2, fy2 = figure
    # 侧图：文字主要在另一侧，全部放 before，图单独一块
    # 用竖直范围判断是否为「行间插图」
    before, after = [], []
    for L in lines:
        if L.cy < fy1:
            before.append(L)
        elif L.cy > fy2:
            after.append(L)
        else:
            # 与图同一水平带：放在图左侧的文字归 before
            if L.cx < fx1:
                before.append(L)
            elif L.cx > fx2:
                after.append(L)
    return before, after


def _scale_figure(fig_clean: Image.Image, col_w: int) -> Image.Image:
    fw, fh = fig_clean.size
    scale = min(col_w / max(1, fw), 1.5)
    fig_clean = fig_clean.resize(
        (max(1, int(fw * scale)), max(1, int(fh * scale))),
        Image.LANCZOS,
    )
    if fig_clean.width < col_w:
        pad = Image.new("RGB", (col_w, fig_clean.height), "white")
        pad.paste(fig_clean, ((col_w - fig_clean.width) // 2, 0))
        return pad
    return fig_clean


def split_question_bands(
    lines: List[OcrLine], h: int, pad: int = 6
) -> Optional[List[Tuple[int, int, List[OcrLine]]]]:
    """按题号把整页 OCR 切成若干竖直带。题号不足 2 个则不切。"""
    starts: List[Tuple[int, OcrLine]] = []
    for L in lines:
        if _RE_QNUM.match((L.text or "").strip()):
            starts.append((L.y1, L))
    if len(starts) < 2:
        return None
    starts.sort(key=lambda x: x[0])
    bands: List[Tuple[int, int, List[OcrLine]]] = []
    for i, (y, _) in enumerate(starts):
        y1 = max(0, y - pad)
        if i + 1 < len(starts):
            y2 = max(y1 + 20, starts[i + 1][0] - pad)
        else:
            y2 = h
        band_lines = [L for L in lines if y1 - 2 <= L.cy < y2 + 2]
        if not band_lines:
            band_lines = [starts[i][1]]
        bands.append((y1, min(h, y2), band_lines))
    return bands


def remap_lines_y(lines: List[OcrLine], y_off: int) -> List[OcrLine]:
    """把行坐标平移到子图坐标系。"""
    out = []
    for L in lines:
        out.append(
            OcrLine(
                text=L.text,
                confidence=L.confidence,
                x1=L.x1,
                y1=L.y1 - y_off,
                x2=L.x2,
                y2=L.y2 - y_off,
            )
        )
    return out


def is_compare_layout_garbled(texts: List[str]) -> bool:
    """比大小题 OCR 是否已乱到不适合直接原样排版（需结构化重建）。"""
    joined = "".join(texts)
    if "比大小" not in joined and not (
        "分米" in joined and ("○" in joined or "米" in joined)
    ):
        if joined.count("○") < 2 and "比" not in joined:
            return False
    if re.search(r"[\^≥≤≈]|20\^|1/2\s*1/7|2\s*[×xX]\s*20", joined):
        return True
    circles = joined.count("○")
    if "比大小" in joined and circles < 2:
        return True
    if "比大小" in joined and re.search(r"\d+\s*[×xX*＊]\s*\d+\s*$", joined):
        if circles <= 1:
            return True
    issues = audit_question_texts(texts)
    if any(x.code == "incomplete_compare" for x in issues) and "比大小" in joined:
        return True
    return False


def rebuild_compare_question(texts: List[str]) -> Optional[List[str]]:
    """
    仅用于「比大小」类题：从 OCR 抽出对照项，楷体重排。
    不做跨试卷猜题；六年级「在○内填」不走此路径。
    """
    cleaned = [normalize_math_text(t or "") for t in texts if (t or "").strip()]
    if not cleaned:
        return None
    joined = " ".join(cleaned)
    if looks_like_circle_fill_compare(joined):
        return None
    if not looks_like_elementary_compare(joined) and not (
        "○" in joined and re.search(r"\d+\s*[＋+×xX*＊]\s*\d+", joined)
        and "分米" in joined
    ):
        # 允许极短单框：仅当同时有 ○ 与两侧数字
        if not (
            "○" in joined
            and re.search(
                r"\d.+\s*○\s*\d",
                joined,
            )
            and len(joined) < 48
        ):
            return None

    m = re.search(r"(\d{1,3}\s*[\.．、\)）]\s*)?比大小\s*[:：]?", joined)
    head = (m.group(0).strip() if m else "比大小")
    if not head.endswith(("：", ":")):
        head += "："

    pairs: List[Tuple[str, str]] = []

    def add_pair(a: str, b: str) -> None:
        a, b = re.sub(r"\s+", "", a), re.sub(r"\s+", "", b)
        if not a or not b:
            return
        if re.search(r"[\^≥≤≈]", a + b):
            return
        if (a, b) in pairs:
            return
        pairs.append((a, b))

    def is_mul_left(num: str) -> bool:
        return bool(re.search(rf"(?<![0-9]){re.escape(num)}\s*[×xX*＊]", joined))

    def normalize_add_left(left: str) -> str:
        if len(left) == 4 and left.startswith("2"):
            return left[1:]
        return left

    um = re.search(
        r"(\d+)\s*分米\D{0,20}?(\d+)\s*(?:米|(?=[×xX*＊○]))", joined
    )
    if um:
        add_pair(f"{um.group(1)}分米", f"{um.group(2)}米")

    mul_lefts: set = set()
    mul_rights: set = set()
    xm_circ = re.search(
        r"(\d{2,4})\s*[×xX*＊]\s*(\d{1,3})\s*○\s*(\d{3,5})", joined
    )
    if xm_circ and len(xm_circ.group(1)) >= 2:
        mul_lefts.add(xm_circ.group(1))
        mul_rights.add(xm_circ.group(3))
        add_pair(f"{xm_circ.group(1)}×{xm_circ.group(2)}", xm_circ.group(3))
    else:
        xm = re.search(r"(\d{2,4})\s*[×xX*＊]\s*(\d{1,3})(?!\d)", joined)
        if xm and len(xm.group(1)) >= 2:
            left_a, left_b = xm.group(1), xm.group(2)
            mul_lefts.add(left_a)
            tail = joined[xm.end() :]
            rights = re.findall(r"(?<![0-9/])(\d{3,5})(?![0-9/])", tail)
            rights = [r for r in rights if r not in {left_a, left_b}]
            rights4 = [r for r in rights if len(r) >= 4]
            pick = rights4[0] if rights4 else (rights[0] if rights else None)
            if pick:
                mul_rights.add(pick)
                add_pair(f"{left_a}×{left_b}", pick)
        else:
            xm_tail = re.search(r"[×xX*＊]\s*(\d{1,3})\s*○\s*(\d{3,5})", joined)
            am_bad = re.search(r"[＋+]\s*\d{2,4}\s*○\s*(\d{2,4})", joined)
            if xm_tail and am_bad:
                left_a = am_bad.group(1)
                if len(left_a) >= 2 and left_a not in mul_lefts:
                    mul_lefts.add(left_a)
                    mul_rights.add(xm_tail.group(2))
                    add_pair(f"{left_a}×{xm_tail.group(1)}", xm_tail.group(2))

    am_circ = re.search(
        r"(\d{2,4})\s*[＋+]\s*(\d{2,4})\s*○\s*(\d{3,5})", joined
    )
    add_ops: set = set()
    if am_circ:
        raw_left = am_circ.group(1)
        left = normalize_add_left(raw_left)
        right = am_circ.group(3)
        add_ops = {left, raw_left, am_circ.group(2)}
        if right not in mul_lefts and right not in mul_rights and not is_mul_left(right):
            add_pair(f"{left}+{am_circ.group(2)}", right)
        else:
            # 右侧被下一题粘走：只使用文本里真实出现的对比数，不臆造
            span = joined[am_circ.start() : (xm_circ.start() if xm_circ else len(joined))]
            rights = re.findall(r"(?<![0-9/])(\d{3,5})(?![0-9/])", span)
            for r in rights:
                if (
                    r not in add_ops
                    and r not in mul_lefts
                    and r not in mul_rights
                    and not is_mul_left(r)
                ):
                    add_pair(f"{left}+{am_circ.group(2)}", r)
                    break
    else:
        am = re.search(r"(\d{2,4})\s*[＋+]\s*(\d{2,4})", joined)
        if am and "○" in joined:
            left = normalize_add_left(am.group(1))
            add_ops = {left, am.group(2)}
            tail = joined[am.end() :]
            rights = re.findall(r"(?<![0-9/])(\d{3,5})(?![0-9/])", tail)
            for r in rights:
                if r in mul_lefts or r in mul_rights or is_mul_left(r) or r in add_ops:
                    continue
                add_pair(f"{left}+{am.group(2)}", r)
                break

    fm = re.search(r"(\d+)\s*/\s*(\d+)\s*○\s*(\d+)\s*/\s*(\d+)", joined)
    if fm:
        add_pair(f"{fm.group(1)}/{fm.group(2)}", f"{fm.group(3)}/{fm.group(4)}")

    if len(pairs) < 4:
        for a, b in re.findall(
            r"(\d+(?:\s*/\s*\d+)?(?:\s*[＋+\-−×xX*＊]\s*\d+(?:\s*/\s*\d+)?)?\s*(?:分米|厘米|毫米|米)?)"
            r"\s*○\s*"
            r"(\d+(?:\s*/\s*\d+)?(?:\s*[＋+\-−×xX*＊]\s*\d+(?:\s*/\s*\d+)?)?\s*(?:分米|厘米|毫米|米)?)",
            joined,
        ):
            if len(pairs) >= 4:
                break
            aa, bb = re.sub(r"\s+", "", a), re.sub(r"\s+", "", b)
            if ("+" in aa or "＋" in aa) and is_mul_left(bb):
                continue
            add_pair(aa, bb)

    pairs = [
        (a, b)
        for a, b in pairs
        if not (("+" in a or "＋" in a) and is_mul_left(b))
    ]

    if len(pairs) < 1:
        return None
    # 没有「比大小」题干时，至少要有完整 ○ 对，禁止臆造右侧
    if "比大小" not in joined and len(pairs) < 1:
        return None

    def sort_key(p):
        a, _ = p
        if "分米" in a:
            return 0
        if "+" in a or "＋" in a:
            return 1
        if "×" in a:
            return 2
        if "/" in a:
            return 3
        return 4

    pairs = sorted(pairs, key=sort_key)

    if "比大小" in joined or len(pairs) >= 2:
        out = [head] if "比大小" in joined else []
        if "比大小" in joined and not out[0].endswith(("：", ":")):
            out[0] += "："
        row: List[str] = []
        for a, b in pairs[:4]:
            row.append(f"{a} ○ {b}")
            if len(row) == 2:
                out.append("    ".join(row))
                row = []
        if row:
            out.append("    ".join(row))
        return out if out else None

    a, b = pairs[0]
    return [f"{a} ○ {b}"]


def format_compare_grid(texts: List[str]) -> List[str]:
    """比大小已较清晰时，整理成每行至多两项；否则原样返回。"""
    cleaned = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        return texts
    joined = " ".join(cleaned)
    if "比大小" not in joined:
        return texts
    if sum(t.count("○") for t in cleaned) >= 3 and len(cleaned) >= 2:
        return cleaned
    rebuilt = rebuild_compare_question(cleaned)
    if rebuilt:
        return rebuilt
    return cleaned


def render_one_question_block(
    crop: Image.Image,
    lines: List[OcrLine],
    cfg: dict,
    erase_fn,
    col_w: int,
    font_path: str,
    font_size: int,
    draft_lines: int,
) -> Tuple[Image.Image, str]:
    """处理单题（或已切好的一题带）：返回 (块图, 模式)。"""
    w, h = crop.size
    texts = dedupe_ocr_texts([normalize_math_text(L.text) for L in lines])
    joined = " ".join(texts)

    def _erase():
        erase_cfg = dict(cfg)
        erase_cfg["grayscale"] = 0
        return erase_fn(crop, erase_cfg), "fallback"

    # 比例尺 / 示意图：不走楷体硬排
    if looks_like_scale_or_diagram(joined):
        return _erase()

    # 六年级填○比较：OCR 崩了就擦除，禁止用三年级启发式乱补
    if is_poor_circle_fill_ocr(texts):
        return _erase()

    # 仅「比大小」类才结构化重建
    if looks_like_elementary_compare(joined) or is_compare_layout_garbled(texts):
        rebuilt = rebuild_compare_question(texts)
        if rebuilt and not is_broken_compare_ocr(rebuilt):
            img = render_question_text(
                rebuilt,
                col_w,
                font_path,
                font_size=font_size,
                draft_lines=draft_lines,
            )
            return img, "text_only"
        if is_broken_compare_ocr(texts) or (
            looks_like_elementary_compare(joined) and rebuilt is None
        ):
            return _erase()

    mode = classify_layout(lines, w, h, cfg, src_img=crop) if lines else "fallback"
    if mode == "fallback" or not lines:
        return _erase()

    fig = None
    if mode == "mixed" or (
        _RE_FIGURE_CUE.search("".join(texts)) and crop is not None
    ):
        fig = find_figure_bbox(
            lines,
            w,
            h,
            min_h_ratio=float(cfg.get("figure_min_h_ratio", 0.12)),
            min_w_ratio=float(cfg.get("figure_min_w_ratio", 0.22)),
            min_area_ratio=float(cfg.get("figure_min_area_ratio", 0.04)),
            src_img=crop,
        )
        if fig is None and _RE_FIGURE_CUE.search("".join(texts)):
            fig = find_figure_by_cue(lines, w, h, crop)
        if fig is not None:
            mode = "mixed"
        elif mode == "mixed":
            mode = "text_only"

    if mode == "mixed" and fig is not None:
        text_lines = lines_outside_figure(lines, fig)
        before, after = split_lines_by_figure(text_lines, fig)
        blocks: List[Image.Image] = []
        before_txt = format_compare_grid(
            dedupe_ocr_texts([L.text for L in before])
        ) if before else []
        after_txt = format_compare_grid(
            dedupe_ocr_texts([L.text for L in after])
        ) if after else []
        if before_txt:
            blocks.append(
                render_question_text(
                    before_txt,
                    col_w,
                    font_path,
                    font_size=font_size,
                    draft_lines=0,
                )
            )
        fx1, fy1, fx2, fy2 = fig
        fig_img = crop.crop((fx1, fy1, fx2, fy2))
        erase_cfg = dict(cfg)
        erase_cfg["grayscale"] = 0
        fig_clean = erase_fn(fig_img, erase_cfg)
        blocks.append(_scale_figure(fig_clean, col_w))
        if after_txt:
            blocks.append(
                render_question_text(
                    after_txt,
                    col_w,
                    font_path,
                    font_size=font_size,
                    draft_lines=0,
                )
            )
        if not blocks:
            return _erase()
        return assemble_blocks(blocks, col_w, gap=int(cfg.get("block_gap", 24))), "mixed"

    texts = format_compare_grid(texts)
    texts = dedupe_ocr_texts(texts)
    img = render_question_text(
        texts, col_w, font_path, font_size=font_size, draft_lines=draft_lines
    )
    return img, "text_only"


def process_question(
    crop: Image.Image,
    cfg: dict,
    client,
    erase_fn,
    col_w: int,
    jpeg_quality: int = 95,
) -> Tuple[Image.Image, str]:
    """
    处理单题，返回 (题块图, 模式标签)。
    模式：text_only | mixed | fallback
    若一框含多道题，按题号切开分别处理再拼回（避免整页侧图丢失、比大小串行）。
    """
    # 先擦除手写，再识别印刷题干；不根据括号或数字猜测答案。
    clean_cfg = dict(cfg)
    clean_cfg["grayscale"] = 0
    crop = erase_fn(crop, clean_cfg)
    # 默认保留擦除后的整块图片。数学公式、分数、表格和跨行配图不能由
    # OCR 启发式可靠复原；完整保留图像比重排后改变题意更安全。
    if cfg.get("mode", "hybrid") != "typeset":
        return crop.convert("RGB"), "image_clean"

    # 实验性的 typeset 模式才继续 OCR 重排。后续均来自已擦除图，
    # 不重复调用云端擦除。
    erase_fn = lambda image, config: image.convert("RGB")
    font_path = find_kaiti_font(cfg.get("kaiti_font", ""))
    font_size = int(cfg.get("font_size", 42))
    draft_lines = int(cfg.get("draft_lines", 3))
    w, h = crop.size

    try:
        raw_lines = ocr_lines(client, crop, jpeg_quality)
    except Exception:
        raw_lines = []

    lines = clean_ocr_lines(raw_lines, min_conf=float(cfg.get("ocr_line_min_conf", 55)))
    if not lines:
        erased = erase_fn(crop, cfg)
        return erased, "fallback"

    bands = split_question_bands(lines, h)
    # 多题同框：逐题处理
    if bands and len(bands) >= 2:
        blocks: List[Image.Image] = []
        modes: List[str] = []
        q_gap = max(24, int(cfg.get("draft_gap", 200) * 0.35))
        for y1, y2, blines in bands:
            # 略加边距，避免切掉配图下沿
            top = max(0, y1 - 4)
            bot = min(h, y2 + 8)
            sub = crop.crop((0, top, w, bot))
            sub_lines = remap_lines_y(blines, top)
            block, mode = render_one_question_block(
                sub,
                sub_lines,
                cfg,
                erase_fn,
                col_w,
                font_path,
                font_size,
                draft_lines=0,
            )
            blocks.append(block)
            modes.append(mode)
        if not blocks:
            return erase_fn(crop, cfg), "fallback"
        # 题间留草稿空隙
        spaced: List[Image.Image] = []
        for i, b in enumerate(blocks):
            spaced.append(b)
            if i < len(blocks) - 1:
                spaced.append(Image.new("RGB", (col_w, q_gap), "white"))
        # 汇总模式
        if any(m == "mixed" for m in modes):
            label = "mixed"
        elif all(m == "fallback" for m in modes):
            label = "fallback"
        else:
            label = "text_only"
        return assemble_blocks(spaced, col_w, gap=0), label

    return render_one_question_block(
        crop,
        lines,
        cfg,
        erase_fn,
        col_w,
        font_path,
        font_size,
        draft_lines,
    )
