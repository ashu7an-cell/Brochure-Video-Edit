#!/usr/bin/env python3
"""
redact_phones.py — Detect phone numbers in a PDF brochure and visually
hide them by painting a background-colored patch over them.

Provides both:
  - process_pdf_bytes(): For in-memory Streamlit / API workflows.
  - process_pdf(): For CLI / local file workflows.
"""
import sys
import io
import re

try:
    import pymupdf as fitz
except ImportError:
    import fitz

import numpy as np
from PIL import Image

try:
    import os as _os
    import pytesseract

    _tesseract_cmd = _os.environ.get("TESSERACT_CMD")
    if _tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

PHONE_RE = re.compile(
    r'(?<!\d)(?:(?:\+?91)[\s\-]?)?([6-9](?:[\s\-]?\d){9})(?!\d)',
    re.IGNORECASE,
)
MIN_DIGIT_COUNT = 10
MAX_DIGIT_COUNT = 12

EMAIL_OR_URL_RE = re.compile(
    r'(@|www\.|https?://|\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b|\b[a-z0-9.-]+\.(?:in|com|org|net|co|io)\b)',
    re.IGNORECASE
)

PHONE_LABEL_WORDS = (
    "ph", "tel", "telephone", "phone", "mobile", "mob", "cell", "fax",
    "contact details", "contact information", "contact no", "contact number",
    "call for booking", "call for bookings", "call now", "call us",
    "call for enquiry", "call for enquiries",
    "for booking", "for bookings", "book now",
    "booking enquiry", "booking enquiries", "for enquiry", "for enquiries",
    "call for details", "call for site visit", "for site visit",
)
_LABEL_ALTERNATION = '|'.join(
    re.escape(w) for w in sorted(PHONE_LABEL_WORDS, key=len, reverse=True)
)

_LABEL_ANYWHERE_RE = re.compile(
    rf'\b(?:{_LABEL_ALTERNATION})\b[:.\-]?', re.IGNORECASE,
)

_CALL_ACTION_ANYWHERE_RE = re.compile(
    r'\bcall\b[\w\s/&,]{0,30}\b(?:book(?:ing)?s?|enquir(?:y|ies))\b\s*[:.\-]?',
    re.IGNORECASE,
)

HEADING_ONLY_RE = re.compile(
    rf'^\s*(?:{_LABEL_ALTERNATION})\s*[:.\-]?\s*$', re.IGNORECASE,
)
_DIGIT_RUN_RE = re.compile(r'\d[\d\s\-]{4,}\d')

WORD_GAP = 4


def _union_rect(rects):
    return fitz.Rect(
        min(r.x0 for r in rects), min(r.y0 for r in rects),
        max(r.x1 for r in rects), max(r.y1 for r in rects),
    )


def _contiguous_runs(idxs):
    if not idxs:
        return []
    idxs = sorted(idxs)
    runs, run = [], [idxs[0]]
    for i in idxs[1:]:
        if i == run[-1] + 1:
            run.append(i)
        else:
            runs.append(run)
            run = [i]
    runs.append(run)
    return runs


def _line_concat_and_offsets(line_words):
    concat = ""
    offsets = []
    for i, (raw, disp, text) in enumerate(line_words):
        start = len(concat)
        concat += text
        offsets.append((start, len(concat), i))
        concat += " "
    return concat, offsets


def _char_span_to_word_span(s, e, offsets):
    word_idxs = [wi for (ws, we, wi) in offsets if ws < e and we > s]
    if not word_idxs:
        return None
    return (min(word_idxs), max(word_idxs))


def find_label_and_number_spans(line_words):
    concat, offsets = _line_concat_and_offsets(line_words)

    email_url_word_idxs = set()
    for m in EMAIL_OR_URL_RE.finditer(concat):
        span = _char_span_to_word_span(m.start(), m.end(), offsets)
        if span:
            email_url_word_idxs.update(range(span[0], span[1] + 1))

    label_spans = []
    for pattern in (_LABEL_ANYWHERE_RE, _CALL_ACTION_ANYWHERE_RE):
        for m in pattern.finditer(concat):
            span = _char_span_to_word_span(m.start(), m.end(), offsets)
            if span:
                label_spans.append(span)

    digit_spans = []
    for m in _DIGIT_RUN_RE.finditer(concat):
        if sum(ch.isdigit() for ch in m.group(0)) < 6:
            continue
        span = _char_span_to_word_span(m.start(), m.end(), offsets)
        if span:
            digit_spans.append(span)

    matches = []
    used_label_spans = set()

    for (lmin, lmax) in label_spans:
        best = None
        best_gap = None
        for (dmin, dmax) in digit_spans:
            gap = dmin - lmax - 1 if dmin > lmax else (lmin - dmax - 1 if lmin > dmax else 0)
            if gap <= WORD_GAP and (best_gap is None or gap < best_gap):
                best, best_gap = (dmin, dmax), gap

        if best:
            dmin, dmax = best

            label_rects = [line_words[i][0] for i in range(lmin, lmax + 1)]
            digit_rects = [line_words[i][0] for i in range(dmin, dmax + 1)]
            label_box = _union_rect(label_rects)
            digit_box = _union_rect(digit_rects)
            heights = [r.height for r in label_rects + digit_rects if r.height > 0]
            avg_h = sum(heights) / len(heights) if heights else 10

            gap_x = max(0, max(label_box.x0, digit_box.x0) - min(label_box.x1, digit_box.x1))
            gap_y = max(0, max(label_box.y0, digit_box.y0) - min(label_box.y1, digit_box.y1))

            if gap_x > 8.0 * avg_h or gap_y > 3.0 * avg_h:
                continue

            target_idxs = (set(range(lmin, lmax + 1)) | set(range(dmin, dmax + 1))) - email_url_word_idxs

            for run in _contiguous_runs(target_idxs):
                run_rects = [line_words[i][0] for i in run if i < len(line_words)]
                if run_rects:
                    matches.append((_union_rect(run_rects), "[labeled phone line]"))

            used_label_spans.add((lmin, lmax))

    orphan_label_spans = [s for s in label_spans if s not in used_label_spans]
    return matches, orphan_label_spans


def get_words_with_display_coords(page):
    rot = page.rotation_matrix
    out = []
    for (x0, y0, x1, y1, text, block_no, line_no, word_no) in page.get_text("words"):
        raw = fitz.Rect(x0, y0, x1, y1)
        disp = raw * rot
        dx0, dx1 = sorted((disp.x0, disp.x1))
        dy0, dy1 = sorted((disp.y0, disp.y1))
        out.append((raw, fitz.Rect(dx0, dy0, dx1, dy1), text, block_no, line_no))
    return out


def get_words_via_ocr(page, zoom, pix=None):
    if not OCR_AVAILABLE:
        return []

    if pix is None:
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))

    pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    deroti = page.derotation_matrix

    out = []
    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        if not text:
            continue
        conf = data.get('conf', ['0'] * n)[i]
        try:
            if float(conf) < 0:
                continue
        except (ValueError, TypeError):
            pass

        left, top, w, h = (data['left'][i], data['top'][i],
                            data['width'][i], data['height'][i])
        dx0, dy0 = left / zoom, top / zoom
        dx1, dy1 = (left + w) / zoom, (top + h) / zoom
        disp = fitz.Rect(dx0, dy0, dx1, dy1)
        raw = disp * deroti
        rx0, rx1 = sorted((raw.x0, raw.x1))
        ry0, ry1 = sorted((raw.y0, raw.y1))
        block_no = data['block_num'][i]
        par_no = data['par_num'][i]
        line_no = (par_no, data['line_num'][i])
        out.append((fitz.Rect(rx0, ry0, rx1, ry1), disp, text, block_no, line_no))
    return out


def group_lines(words):
    lines = {}
    for (raw, disp, text, block_no, line_no) in words:
        lines.setdefault((block_no, line_no), []).append((raw, disp, text))
    line_list = []
    for key, ws in lines.items():
        ws.sort(key=lambda t: t[1].x0)
        line_list.append(ws)
    return line_list


def find_phone_matches(line_words):
    concat, offsets = _line_concat_and_offsets(line_words)

    matches = []
    for m in PHONE_RE.finditer(concat):
        matched = m.group(0)
        digit_count = sum(ch.isdigit() for ch in matched)
        if digit_count not in (10, 12):
            continue

        span = _char_span_to_word_span(m.start(), m.end(), offsets)
        if span is None:
            continue
        word_idxs = sorted(range(span[0], span[1] + 1))

        rects = [line_words[wi][1] for wi in word_idxs]
        heights = [r.height for r in rects if r.height > 0]
        avg_height = sum(heights) / len(heights) if heights else 10

        if len(rects) > 1:
            for a, b in zip(rects, rects[1:]):
                gap = b.x0 - a.x1
                if gap > 6.0 * avg_height:
                    break
            else:
                gap = None
            if gap is not None and gap > 6.0 * avg_height:
                continue

        raw_rects = [line_words[wi][0] for wi in word_idxs]
        x0 = min(r.x0 for r in raw_rects)
        y0 = min(r.y0 for r in raw_rects)
        x1 = max(r.x1 for r in raw_rects)
        y1 = max(r.y1 for r in raw_rects)

        matches.append((fitz.Rect(x0, y0, x1, y1), matched, (word_idxs[0], word_idxs[-1])))
    return matches


def dedupe_by_overlap(matches):
    deduped = []
    for rect, matched in matches:
        overlapping = [
            i for i, (existing, _existing_matched) in enumerate(deduped)
            if (rect & existing).get_area() > 0
            and (rect & existing).get_area() / max(1.0, min(rect.get_area(), existing.get_area())) > 0.5
        ]
        if not overlapping:
            deduped.append((rect, matched))
            continue
        candidates = [(rect, matched)] + [deduped[i] for i in overlapping]
        best = max(candidates, key=lambda rm: rm[0].get_area())
        for i in sorted(overlapping, reverse=True):
            del deduped[i]
        deduped.append(best)
    return deduped


def sample_background_color(pix, raw_rect, page, zoom, pad=6, ring=10):
    disp_rect = raw_rect * page.rotation_matrix
    x0, x1 = sorted((disp_rect.x0, disp_rect.x1))
    y0, y1 = sorted((disp_rect.y0, disp_rect.y1))
    px0, py0, px1, py1 = [int(v * zoom) for v in (x0, y0, x1, y1)]
    W, H = pix.width, pix.height

    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(H, W, pix.n)[:, :, :3]

    outer_pad = int(pad * zoom) + ring
    top_y = max(py0 - outer_pad, 0)
    bottom_y = min(py1 + outer_pad, H - 1)
    left_x = max(px0 - outer_pad, 0)
    right_x = min(px1 + outer_pad, W - 1)
    xs = np.arange(max(px0 - outer_pad, 0), min(px1 + outer_pad, W), 4)
    ys = np.arange(max(py0 - outer_pad, 0), min(py1 + outer_pad, H), 4)

    pieces = []
    if xs.size:
        pieces.append(arr[top_y, xs])
        pieces.append(arr[bottom_y, xs])
    if ys.size:
        pieces.append(arr[ys, left_x])
        pieces.append(arr[ys, right_x])

    if not pieces:
        return (0.96, 0.96, 0.93)

    samples = np.concatenate(pieces, axis=0).astype(np.float64)
    r, g, b = (float(v) for v in np.median(samples, axis=0))
    return (r / 255, g / 255, b / 255)


DEFAULT_MAX_SIZE_BYTES = 25 * 1_000_000

_COMPRESSION_LADDER = [
    (85, 2400), (75, 2000), (60, 1600), (45, 1200), (30, 1000), (20, 800),
]


def _collect_image_xrefs(doc):
    mask_xrefs = set()
    for page in doc:
        for img in page.get_images(full=True):
            smask_xref = img[1]
            if smask_xref:
                mask_xrefs.add(smask_xref)

    xref_to_page = {}
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            smask_xref = img[1]
            if xref in mask_xrefs or smask_xref:
                continue
            xref_to_page.setdefault(xref, page.number)
    return xref_to_page


def _recompress_image_bytes(original_bytes, quality, max_dim):
    try:
        im = Image.open(io.BytesIO(original_bytes))
        im.load()
    except Exception:
        return None

    if im.format not in ("JPEG", "MPO"):
        return None

    has_alpha = False

    if max_dim and max(im.size) > max_dim:
        ratio = max_dim / max(im.size)
        new_size = (max(1, round(im.width * ratio)), max(1, round(im.height * ratio)))
        im = im.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    try:
        if has_alpha:
            im.convert("RGBA").save(buf, format="PNG", optimize=True)
        else:
            jpeg_mode = "CMYK" if im.mode == "CMYK" else "RGB"
            im.convert(jpeg_mode).save(buf, format="JPEG", quality=quality, optimize=True)
    except Exception:
        return None
    return buf.getvalue()


def _visual_compression_ok(before_bytes, after_bytes, zoom=0.15,
                            max_mean_diff=7.5, max_changed_fraction=0.16):
    try:
        before = fitz.open(stream=before_bytes, filetype="pdf")
        after = fitz.open(stream=after_bytes, filetype="pdf")
        if len(before) != len(after):
            return False

        for i in range(len(before)):
            pb = before[i].get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                                      alpha=False, colorspace=fitz.csRGB)
            pa = after[i].get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                                     alpha=False, colorspace=fitz.csRGB)
            if (pb.width, pb.height) != (pa.width, pa.height):
                return False

            a = np.frombuffer(pb.samples, dtype=np.uint8).astype(np.int16)
            b = np.frombuffer(pa.samples, dtype=np.uint8).astype(np.int16)
            diff = np.abs(a - b)
            mean_diff = float(diff.mean())
            changed_fraction = float(np.mean(np.max(diff.reshape(-1, 3), axis=1) > 25))

            if mean_diff > max_mean_diff or changed_fraction > max_changed_fraction:
                return False

        return True
    except Exception:
        return False


def shrink_pdf_to_size(doc, max_size_bytes=DEFAULT_MAX_SIZE_BYTES):
    baseline_bytes = doc.tobytes(garbage=4, deflate=True)
    if len(baseline_bytes) <= max_size_bytes:
        return baseline_bytes

    base_doc = fitz.open(stream=baseline_bytes, filetype="pdf")
    xref_to_page = _collect_image_xrefs(base_doc)

    originals = {}
    for xref in xref_to_page:
        try:
            ftype, fvalue = base_doc.xref_get_key(xref, "Filter")
            if ftype == "array" and "/DCTDecode" in fvalue:
                raw = base_doc.xref_stream_raw(xref)
            elif ftype == "name" and fvalue == "/DCTDecode":
                raw = base_doc.xref_stream_raw(xref)
            else:
                continue
            if raw:
                originals[xref] = raw
        except Exception:
            continue

    best_bytes = baseline_bytes

    for quality, max_dim in _COMPRESSION_LADDER:
        trial = fitz.open(stream=baseline_bytes, filetype="pdf")

        for xref, page_no in xref_to_page.items():
            src = originals.get(xref)
            if not src:
                continue

            new_bytes = _recompress_image_bytes(src, quality, max_dim)

            if not new_bytes or len(new_bytes) >= len(src):
                continue

            try:
                trial[page_no].replace_image(xref, stream=new_bytes)
            except Exception:
                continue

        candidate = trial.tobytes(garbage=4, deflate=True)

        if len(candidate) >= len(best_bytes):
            continue

        if not _visual_compression_ok(baseline_bytes, candidate):
            continue

        best_bytes = candidate

        if len(best_bytes) <= max_size_bytes:
            break

    return best_bytes


def process_pdf_bytes(pdf_bytes: bytes, zoom: float = 2.0, pad: float = 2.0) -> bytes:
    """Process a PDF directly from memory bytes and return modified PDF bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_idx, page in enumerate(doc):
        words = get_words_with_display_coords(page)
        if not words:
            words = get_words_via_ocr(page, zoom)

        if not words:
            continue

        lines = group_lines(words)
        page_matches = []

        for line in lines:
            matches = find_phone_matches(line)
            for rect, matched, _ in matches:
                page_matches.append((rect, matched))

            lbl_matches, _ = find_label_and_number_spans(line)
            page_matches.extend(lbl_matches)

        deduped = dedupe_by_overlap(page_matches)
        if not deduped:
            continue

        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))

        for rect, _ in deduped:
            color = sample_background_color(pix, rect, page, zoom)
            padded_rect = fitz.Rect(
                rect.x0 - pad, rect.y0 - pad,
                rect.x1 + pad, rect.y1 + pad
            )
            shape = page.new_shape()
            shape.draw_rect(padded_rect)
            shape.finish(fill=color, color=None)
            shape.commit()

    return shrink_pdf_to_size(doc)


def process_pdf(input_path: str, output_path: str, zoom: float = 2.0, pad: float = 2.0):
    """File path wrapper for CLI usage."""
    with open(input_path, "rb") as f:
        input_bytes = f.read()

    output_bytes = process_pdf_bytes(input_bytes, zoom=zoom, pad=pad)

    with open(output_path, "wb") as f:
        f.write(output_bytes)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 redact_phones.py input.pdf output.pdf")
        sys.exit(1)
    process_pdf(sys.argv[1], sys.argv[2])
