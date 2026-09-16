#!/usr/bin/env python3
"""
redact_phones.py — Detect phone numbers in a PDF brochure and visually
hide them by painting a background-colored patch over them.

Tasks:
  1. Detect and visually redact all mobile and landline phone numbers.
  2. Compress the file to stay within 25MB without removing or corrupting
     any architectural images, graphics, or content.
"""

import sys
import io
import re
import os as _os
import numpy as np
from PIL import Image

try:
    import pymupdf as fitz
except ImportError:
    import fitz

try:
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

PHONE_LABEL_WORDS = (
    "ph", "tel", "telephone", "phone", "mobile", "mob", "cell", "fax",
    "contact details", "contact information", "contact no", "contact number",
    "call for booking", "call for bookings", "call now", "call us",
    "for booking", "for bookings", "book now",
    "booking enquiry", "booking enquiries", "for enquiry", "for enquiries",
    "call for details", "call for site visit", "for site visit",
)

_LABEL_ALTERNATION = '|'.join(
    re.escape(w) for w in sorted(PHONE_LABEL_WORDS, key=len, reverse=True)
)

HEADING_ONLY_RE = re.compile(
    rf'^\s*(?:{_LABEL_ALTERNATION})\s*[:.\-]?\s*$', re.IGNORECASE,
)

_CALL_ACTION_RE = re.compile(
    r'^\s*call\b[\w\s/&,]{0,30}\b(?:book(?:ing)?s?|enquir(?:y|ies))\b\s*[:.\-]?\s*$',
    re.IGNORECASE,
)

_LABEL_PREFIX_RE = re.compile(
    rf'^\s*(?:{_LABEL_ALTERNATION})\s*[:.\-]?\s*', re.IGNORECASE,
)
_DIGITS_AND_SEPARATORS_RE = re.compile(r'^[\d\s,+\-/]+$')
_DIGIT_RUN_RE = re.compile(r'\d[\d\s\-]{4,}\d')

DEFAULT_MAX_SIZE_BYTES = 25 * 1_000_000

# Mild compression ladder - stops immediately if quality degrades drastically
_COMPRESSION_LADDER = [
    (85, 2400), (75, 2000), (65, 1600)
]


def _union_rect(rects):
    return fitz.Rect(
        min(r.x0 for r in rects), min(r.y0 for r in rects),
        max(r.x1 for r in rects), max(r.y1 for r in rects),
    )


def find_label_or_heading_match(line_words):
    if not line_words:
        return None
    concat = " ".join(text for (_raw, _disp, text) in line_words).strip()
    if not concat:
        return None

    if HEADING_ONLY_RE.match(concat) or _CALL_ACTION_RE.match(concat):
        return (_union_rect([raw for (raw, _disp, _text) in line_words]), True)

    m = _LABEL_PREFIX_RE.match(concat)
    if m:
        remainder = concat[m.end():].strip()
        if remainder and _DIGITS_AND_SEPARATORS_RE.match(remainder) and _DIGIT_RUN_RE.search(remainder):
            return (_union_rect([raw for (raw, _disp, _text) in line_words]), False)

    return None


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
    concat = ""
    offsets = []
    for i, (raw, disp, text) in enumerate(line_words):
        start = len(concat)
        concat += text
        offsets.append((start, len(concat), i))
        concat += " "

    matches = []
    for m in PHONE_RE.finditer(concat):
        matched = m.group(0)
        digit_count = sum(ch.isdigit() for ch in matched)
        if digit_count not in (10, 12):
            continue

        s, e = m.start(), m.end()
        word_idxs = [wi for (ws, we, wi) in offsets if ws < e and we > s]
        if not word_idxs:
            continue

        word_idxs = sorted(word_idxs)
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

        matches.append((fitz.Rect(x0, y0, x1, y1), matched))
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


def _collect_image_xrefs(doc):
    """Safely collects image xrefs while strictly protecting soft masks and alpha channels."""
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
            # EXCLUDE soft masks completely to avoid invisible/missing images
            if xref in mask_xrefs:
                continue
            xref_to_page.setdefault(xref, page.number)
    return xref_to_page


def _recompress_image_bytes(original_bytes, quality, max_dim):
    try:
        im = Image.open(io.BytesIO(original_bytes))
        im.load()
    except Exception:
        return None

    # SKIP any image with transparency/alpha channels to prevent dropped assets
    if im.mode in ("RGBA", "LA", "P") or "transparency" in im.info:
        return None

    if max_dim and max(im.size) > max_dim:
        ratio = max_dim / max(im.size)
        new_size = (max(1, round(im.width * ratio)), max(1, round(im.height * ratio)))
        im = im.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    try:
        im.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    except Exception:
        return None
    return buf.getvalue()


def shrink_pdf_to_size(doc, max_size_bytes=DEFAULT_MAX_SIZE_BYTES):
    """Recompresses non-critical JPEG photos safely. Prints a clear alert if 
    compression cannot reach target limit without removing content."""
    current_bytes = doc.tobytes(garbage=4, deflate=True)
    if len(current_bytes) <= max_size_bytes:
        return current_bytes

    xref_to_page = _collect_image_xrefs(doc)
    originals = {}
    for xref in xref_to_page:
        try:
            originals[xref] = doc.extract_image(xref)["image"]
        except Exception:
            continue

    for quality, max_dim in _COMPRESSION_LADDER:
        for xref, page_no in xref_to_page.items():
            src = originals.get(xref)
            if not src:
                continue
            new_bytes = _recompress_image_bytes(src, quality, max_dim)
            if not new_bytes or len(new_bytes) >= len(src):
                continue
            try:
                doc[page_no].replace_image(xref, stream=new_bytes)
            except Exception:
                continue

        current_bytes = doc.tobytes(garbage=4, deflate=True)
        if len(current_bytes) <= max_size_bytes:
            break

    if len(current_bytes) > max_size_bytes:
        print(f"\n[WARNING]: Output file size ({len(current_bytes) / 1_000_000:.2f} MB) "
              f"exceeds target limit of {max_size_bytes / 1_000_000:.0f} MB. "
              f"No content or images were removed to preserve document integrity.")

    return current_bytes


def _redact_document(doc, pad=4.0, zoom=2):
    total_found = 0

    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)) if OCR_AVAILABLE else None

        text_words = get_words_with_display_coords(page)
        ocr_words = get_words_via_ocr(page, zoom, pix=pix)
        used_ocr = bool(ocr_words)

        page_matches = []
        heading_candidates = []

        for lw in group_lines(text_words) + group_lines(ocr_words):
            page_matches.extend(find_phone_matches(lw))

            label_result = find_label_or_heading_match(lw)
            if label_result:
                rect, is_heading_only = label_result
                if is_heading_only:
                    heading_candidates.append((rect, "[phone heading]"))
                else:
                    page_matches.append((rect, "[labeled phone line]"))

        page_matches = dedupe_by_overlap(page_matches)
        heading_candidates = dedupe_by_overlap(heading_candidates)

        if page_matches and heading_candidates:
            page_matches.extend(heading_candidates)

        if not page_matches:
            continue

        if pix is None:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        shape = page.new_shape()

        for (raw_rect, matched_text) in page_matches:
            total_found += 1
            padded = fitz.Rect(raw_rect.x0 - pad, raw_rect.y0 - pad,
                                raw_rect.x1 + pad, raw_rect.y1 + pad)
            color = sample_background_color(pix, raw_rect, page, zoom, pad=pad)
            src = "OCR" if used_ocr else "text"
            print(f"Page {page.number+1} [{src}]: found '{matched_text}' "
                  f"-> patched with color {tuple(round(c,3) for c in color)}")
            shape.draw_rect(padded)
            shape.finish(fill=color, color=color)

        shape.commit()

    return total_found


def process_pdf(in_path, out_path, pad=4.0, zoom=2, max_size_bytes=DEFAULT_MAX_SIZE_BYTES):
    doc = fitz.open(in_path)
    total_found = _redact_document(doc, pad=pad, zoom=zoom)
    out_bytes = shrink_pdf_to_size(doc, max_size_bytes=max_size_bytes)
    with open(out_path, "wb") as f:
        f.write(out_bytes)
    print(f"\nDone. {total_found} phone number(s) visually covered. "
          f"Final size: {len(out_bytes) / 1_000_000:.2f} MB. Saved to {out_path}")
    return total_found


def process_pdf_bytes(pdf_bytes, pad=4.0, zoom=2, max_size_bytes=DEFAULT_MAX_SIZE_BYTES):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_found = _redact_document(doc, pad=pad, zoom=zoom)
    out_bytes = shrink_pdf_to_size(doc, max_size_bytes=max_size_bytes)
    print(f"\nDone. {total_found} phone number(s) visually covered. "
          f"Final size: {len(out_bytes) / 1_000_000:.2f} MB.")
    return out_bytes, total_found


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 redact_phones.py input.pdf output.pdf")
        sys.exit(1)
    if not OCR_AVAILABLE:
        print("Warning: pytesseract not installed — pages with no real text "
              "layer will NOT be checked.", file=sys.stderr)
    process_pdf(sys.argv[1], sys.argv[2])
