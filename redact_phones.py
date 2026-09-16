#!/usr/bin/env python3
"""
redact_phones.py — Detect phone numbers in a PDF brochure and visually
hide them by painting a background-colored patch over them (no destructive
content-stream redaction, so it's a pure visual cover-up, not a
guaranteed-unrecoverable redaction).

Two detection paths, chosen per-page automatically:
  1. Real text (page.get_text("words")) — fast, used when the page has an
     actual text layer.
  2. OCR (pytesseract on a rendered pixmap) — used when a page has NO
     extractable text at all, which happens when a PDF is exported from a
     design tool (Illustrator/Canva/InDesign) with fonts converted to
     vector outlines/curves, or when a page is a scanned image. In that
     case there are no text objects for get_text() to find, so phone
     numbers are invisible to path 1 no matter how good the regex is.

Handles pages with a /Rotate entry: detection/grouping happens in the
page's final display orientation, while patches are drawn in the PDF's
raw (pre-rotation) coordinate space, which is what content-drawing
operations expect.

Usage:
    python3 redact_phones.py input.pdf output.pdf
"""
import sys
import io
import re
try:
    # PyMuPDF >= 1.24.3 exposes the importable name "pymupdf".
    import pymupdf as fitz
except ImportError:
    # Older PyMuPDF releases only expose the legacy name "fitz".
    import fitz
import numpy as np
from PIL import Image

try:
    import os as _os

    import pytesseract

    # On Streamlit Cloud (Linux), tesseract is installed via packages.txt
    # ("tesseract-ocr") and lands on PATH as `tesseract`, so pytesseract's
    # default lookup works with no configuration needed. Only override the
    # binary path if TESSERACT_CMD is explicitly set (e.g. for local Windows
    # dev), so this same file works unmodified in both environments.
    _tesseract_cmd = _os.environ.get("TESSERACT_CMD")
    if _tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Indian mobile numbers are normally 10 digits beginning with 6-9.
# Also accept +91 / 91 prefixes and common brochure separators.
# Matching is intentionally performed on OCR/text lines rather than requiring
# the phone number to be a single PDF/OCR word.
PHONE_RE = re.compile(
    r'(?<!\d)(?:(?:\+?91)[\s\-]?)?([6-9](?:[\s\-]?\d){9})(?!\d)',
    re.IGNORECASE,
)
MIN_DIGIT_COUNT = 10
MAX_DIGIT_COUNT = 12  # 10-digit mobile, optionally +91/91

# --- Landline numbers and phone-related headings ----------------------------
# Indian landline numbers (an STD code plus a 6-8 digit local number, often
# several of them listed together after one label, e.g.
# "Tel: 0471 2436173, 2436175, 2436401") don't fit the fixed-width mobile
# pattern above, and their formats vary too much (STD code length, how many
# numbers are listed, whether the STD code repeats) to regex reliably in
# isolation. Landline numbers also almost always appear right after a
# recognizable label ("Tel", "Ph", "Phone", "Mobile", "Contact Details", ...),
# so instead of trying to parse each number out individually, we detect the
# whole line as phone-related and cover it end to end - label and numbers
# together. That also naturally satisfies removing standalone headings like a
# lone "Ph:" line that sits above the actual number line.
PHONE_LABEL_WORDS = (
    "ph", "tel", "telephone", "phone", "mobile", "mob", "cell", "fax",
    "contact details", "contact information", "contact no", "contact number",
    # Real-estate brochures often use a call-to-action banner instead of (or
    # alongside) a plain label - e.g. a "Call for Booking :" row with the
    # number(s) right after it. Without these, that row's heading survives
    # even after the number next to it is redacted, which is exactly the
    # "orphaned heading" look this list is meant to prevent.
    "call for booking", "call for bookings", "call now", "call us",
    "for booking", "for bookings", "book now",
    "booking enquiry", "booking enquiries", "for enquiry", "for enquiries",
    "call for details", "call for site visit", "for site visit",
)
_LABEL_ALTERNATION = '|'.join(
    re.escape(w) for w in sorted(PHONE_LABEL_WORDS, key=len, reverse=True)
)
# A line that is *just* one of the labels (e.g. a standalone "Ph:" heading
# with the real numbers elsewhere) - only redacted if the page also has an
# actual phone-number match somewhere on it.
HEADING_ONLY_RE = re.compile(
    rf'^\s*(?:{_LABEL_ALTERNATION})\s*[:.\-]?\s*$', re.IGNORECASE,
)
# Looser fallback for call-to-action headings phrased in ways the fixed list
# above doesn't cover verbatim (e.g. "Call Us For Booking Now", "Call For
# Booking / Site Visit"). Still anchored on "call" plus "book"/"enquir" so
# it can't casually match an unrelated heading, and - like HEADING_ONLY_RE -
# is only ever used to strip a line, contingent on the page already having
# an actual phone-number match somewhere on it.
_CALL_ACTION_RE = re.compile(
    r'^\s*call\b[\w\s/&,]{0,30}\b(?:book(?:ing)?s?|enquir(?:y|ies))\b\s*[:.\-]?\s*$',
    re.IGNORECASE,
)
# A line that starts with a label and is followed by what looks like one or
# more phone numbers (digits, spaces, +, commas, hyphens - no other words).
_LABEL_PREFIX_RE = re.compile(
    rf'^\s*(?:{_LABEL_ALTERNATION})\s*[:.\-]?\s*', re.IGNORECASE,
)
_DIGITS_AND_SEPARATORS_RE = re.compile(r'^[\d\s,+\-/]+$')
_DIGIT_RUN_RE = re.compile(r'\d[\d\s\-]{4,}\d')  # a loose run of >=6 digits


def _union_rect(rects):
    return fitz.Rect(
        min(r.x0 for r in rects), min(r.y0 for r in rects),
        max(r.x1 for r in rects), max(r.y1 for r in rects),
    )


def find_label_or_heading_match(line_words):
    """Check one OCR/text line for a phone label/heading.

    Returns (raw_rect, is_heading_only) covering the WHOLE line if it is
    either a bare label/heading (e.g. "Ph:") or a label followed by one or
    more numbers (e.g. "Tel: 0471 2436173, 2436175, 2436401"), else None.
    """
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
    """Return every real-text word as (raw_bbox, display_bbox, text, block, line)."""
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
    """OCR the rendered page and return words in the same shape as
    get_words_with_display_coords: (raw_bbox, display_bbox, text, block, line).
    Used as a fallback when a page has no real text layer at all.

    ``pix`` lets the caller pass in a pixmap it already rendered (e.g. for
    background-color sampling) so the page isn't rendered twice."""
    if not OCR_AVAILABLE:
        return []

    if pix is None:
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))

    # Build the PIL image straight from the pixmap's raw RGB buffer instead
    # of round-tripping through a PNG encode (tobytes) + decode (Image.open)
    # - same pixels, no compression work.
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
            if float(conf) < 0:  # tesseract uses -1 for non-text rows
                continue
        except (ValueError, TypeError):
            pass

        left, top, w, h = (data['left'][i], data['top'][i],
                            data['width'][i], data['height'][i])
        # pixel space -> PDF point space (display orientation)
        dx0, dy0 = left / zoom, top / zoom
        dx1, dy1 = (left + w) / zoom, (top + h) / zoom
        disp = fitz.Rect(dx0, dy0, dx1, dy1)
        raw = disp * deroti  # back to raw/content-stream space for drawing
        rx0, rx1 = sorted((raw.x0, raw.x1))
        ry0, ry1 = sorted((raw.y0, raw.y1))
        # Tesseract resets line_num within each paragraph, so block_num alone
        # (or block_num + line_num) is not a unique line key — combine with
        # par_num too, or unrelated columns/paragraphs get merged into one
        # fake "line" and produce false-positive digit runs.
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
        ws.sort(key=lambda t: t[1].x0)  # left-to-right in DISPLAY space
        line_list.append(ws)
    return line_list


def find_phone_matches(line_words):
    """Find Indian mobile numbers in one OCR/text line.

    OCR often splits a phone number such as ``73832 32352`` into two words.
    The old implementation rejected these when the gap between OCR boxes was
    more than 1.5x the character height. Brochure layouts can have much larger
    visual spacing, so we now use the text sequence as the primary signal and
    only reject obviously distant boxes.
    """
    concat = ""
    offsets = []
    for i, (raw, disp, text) in enumerate(line_words):
        # Keep a separator so digits in adjacent OCR words remain separate.
        start = len(concat)
        concat += text
        offsets.append((start, len(concat), i))
        concat += " "

    matches = []
    for m in PHONE_RE.finditer(concat):
        matched = m.group(0)
        digit_count = sum(ch.isdigit() for ch in matched)
        # +91/91 + 10-digit mobile = 12 digits maximum.
        if digit_count not in (10, 12):
            continue

        s, e = m.start(), m.end()
        word_idxs = [wi for (ws, we, wi) in offsets if ws < e and we > s]
        if not word_idxs:
            continue

        # Do not require a tiny OCR gap. Only reject candidates where the
        # boxes are clearly separated into unrelated brochure regions.
        word_idxs = sorted(word_idxs)
        rects = [line_words[wi][1] for wi in word_idxs]
        heights = [r.height for r in rects if r.height > 0]
        avg_height = sum(heights) / len(heights) if heights else 10

        # A phone number can be visually spaced in brochure designs.
        # 6x character height is a safer upper bound while still avoiding
        # combining unrelated columns.
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

        # Slightly enlarge the detection box because OCR boxes can clip
        # ascenders/descenders or the first/last digit.
        matches.append((fitz.Rect(x0, y0, x1, y1), matched))
    return matches


def dedupe_by_overlap(matches):
    """Collapse matches whose boxes overlap by more than half of the
    smaller box's area (e.g. the same phone/heading line found once via the
    real text layer and again via OCR, or a labeled-line box that fully
    contains one or more smaller mobile-regex matches on that same line).

    When several matches overlap, the LARGEST box wins - a labeled line
    like "Call for Booking : 9876543210, 9876543211" should redact the
    whole line (label and BOTH numbers), not just whichever smaller
    number-only match happened to be recorded first."""
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
        # Merge every overlapping existing box (there can be more than one
        # when a big labeled-line box overlaps several small number-only
        # matches on that line) plus the new one, keep only the largest.
        candidates = [(rect, matched)] + [deduped[i] for i in overlapping]
        best = max(candidates, key=lambda rm: rm[0].get_area())
        for i in sorted(overlapping, reverse=True):
            del deduped[i]
        deduped.append(best)
    return deduped


def sample_background_color(pix, raw_rect, page, zoom, pad=6, ring=10):
    """Sample the page background around (but outside) the match, in the
    pixmap's DISPLAY pixel space, to find a fill color that blends in.

    Same sample points and same median as a per-pixel Python loop would
    produce, but pulled straight out of a numpy view of the pixmap buffer
    instead of calling pix.pixel() once per point - much less overhead when
    a page has many matches to patch."""
    disp_rect = raw_rect * page.rotation_matrix
    x0, x1 = sorted((disp_rect.x0, disp_rect.x1))
    y0, y1 = sorted((disp_rect.y0, disp_rect.y1))
    px0, py0, px1, py1 = [int(v * zoom) for v in (x0, y0, x1, y1)]
    W, H = pix.width, pix.height

    # Zero-copy view onto the pixmap's raw RGB buffer.
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


# --- Shrinking oversized PDFs -----------------------------------------------
# Used as a hard upload-size ceiling for the backend this tool feeds into.
# Defined in decimal MB (1,000,000 bytes) rather than binary MiB, since
# that's the smaller/more conservative reading of "25 MB" and guarantees the
# output is under the limit either way it's interpreted on the receiving end.
DEFAULT_MAX_SIZE_BYTES = 25 * 1_000_000

# (JPEG quality, max longest-side in px) tried in order, mild first. Photos
# in real-estate brochures are almost always the bulk of the file size, so
# this only ever touches embedded images - text, vector art, and the
# phone-number redaction patches drawn earlier are never touched.
_COMPRESSION_LADDER = [
    (85, 2400), (75, 2000), (60, 1600), (45, 1200), (30, 1000), (20, 800),
]


def _collect_image_xrefs(doc):
    """One page number per unique image xref (an image can be reused across
    pages; we only need to touch it once via any page that has it)."""
    xref_to_page = {}
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            xref_to_page.setdefault(xref, page.number)
    return xref_to_page


def _recompress_image_bytes(original_bytes, quality, max_dim):
    """Recompress one image's ORIGINAL bytes at a given quality/size cap.
    Returns new bytes, or None if the image can't be safely recompressed
    (caller then just leaves that image alone)."""
    try:
        im = Image.open(io.BytesIO(original_bytes))
        im.load()
    except Exception:
        return None

    has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)

    if max_dim and max(im.size) > max_dim:
        ratio = max_dim / max(im.size)
        new_size = (max(1, round(im.width * ratio)), max(1, round(im.height * ratio)))
        im = im.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    try:
        if has_alpha:
            # Keep transparency (logos, watermarks) - PNG compresses less
            # than JPEG but won't corrupt the alpha channel.
            im.convert("RGBA").save(buf, format="PNG", optimize=True)
        else:
            im.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    except Exception:
        return None
    return buf.getvalue()


def shrink_pdf_to_size(doc, max_size_bytes=DEFAULT_MAX_SIZE_BYTES):
    """Recompress embedded images, in place on the open Document, until the
    saved file is at or under max_size_bytes. No-ops immediately if the
    document is already small enough. Returns the final saved bytes."""
    current_bytes = doc.tobytes(garbage=4, deflate=True)
    if len(current_bytes) <= max_size_bytes:
        return current_bytes

    xref_to_page = _collect_image_xrefs(doc)
    # Cache each image's ORIGINAL bytes once. Every attempt below
    # recompresses from this original, never from a previous attempt's
    # output, so quality loss doesn't compound across rungs of the ladder.
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
            # Never swap in a "recompressed" image that's actually bigger.
            if not new_bytes or len(new_bytes) >= len(src):
                continue
            try:
                doc[page_no].replace_image(xref, stream=new_bytes)
            except Exception:
                continue

        current_bytes = doc.tobytes(garbage=4, deflate=True)
        if len(current_bytes) <= max_size_bytes:
            break

    return current_bytes


def _redact_document(doc, pad=4.0, zoom=2):
    """Run detection + patching over every page of an already-open Document,
    in place. Shared by the path-based and bytes-based entry points below."""
    total_found = 0

    for page in doc:
        # Brochures frequently contain a mixture of real PDF text, vector
        # outlines, and embedded images. A page can therefore have a text
        # layer while the phone number itself exists only as an image/vector.
        # Always run OCR in addition to text extraction.
        #
        # Render the page at most once per page: if OCR is available we need
        # a pixmap for it anyway, so render it up front and reuse the same
        # pixmap later for background-color sampling instead of rendering
        # the page a second time. If OCR isn't available, defer the render
        # until we actually know there's a match to patch (same as before).
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)) if OCR_AVAILABLE else None

        text_words = get_words_with_display_coords(page)
        ocr_words = get_words_via_ocr(page, zoom, pix=pix)
        used_ocr = bool(ocr_words)

        page_matches = []       # (rect, description) - actual phone numbers
        heading_candidates = [] # (rect, description) - bare "Ph:"-style headings

        for lw in group_lines(text_words) + group_lines(ocr_words):
            page_matches.extend(find_phone_matches(lw))

            label_result = find_label_or_heading_match(lw)
            if label_result:
                rect, is_heading_only = label_result
                if is_heading_only:
                    heading_candidates.append((rect, "[phone heading]"))
                else:
                    page_matches.append((rect, "[labeled phone line]"))

        # OCR + text can detect the same phone/heading line twice, and a
        # labeled-line box can fully contain a smaller mobile-regex match on
        # the same line - de-duplicate both lists by box overlap.
        page_matches = dedupe_by_overlap(page_matches)
        heading_candidates = dedupe_by_overlap(heading_candidates)

        # Only strip standalone phone headings (e.g. a lone "Ph:" line, or a
        # "Contact Details"/"Contact Information" heading) when the page
        # actually had a phone number redacted somewhere - a heading word
        # showing up with no nearby number is left alone.
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
    """Path in, path out (CLI usage)."""
    doc = fitz.open(in_path)
    total_found = _redact_document(doc, pad=pad, zoom=zoom)
    out_bytes = shrink_pdf_to_size(doc, max_size_bytes=max_size_bytes)
    with open(out_path, "wb") as f:
        f.write(out_bytes)
    print(f"\nDone. {total_found} phone number(s) visually covered. "
          f"Final size: {len(out_bytes) / 1_000_000:.2f} MB. Saved to {out_path}")
    return total_found


def process_pdf_bytes(pdf_bytes, pad=4.0, zoom=2, max_size_bytes=DEFAULT_MAX_SIZE_BYTES):
    """Bytes in, bytes out - lets callers (e.g. the Streamlit app) work
    entirely in memory instead of writing the upload and the result to disk
    as temp files. If the redacted PDF is over max_size_bytes, embedded
    images are recompressed until it fits (or until the compression ladder
    is exhausted, whichever comes first)."""
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
              "layer (vector-outline or scanned pages) will NOT be checked. "
              "Install with: pip install pytesseract (and the tesseract-ocr binary).",
              file=sys.stderr)
    process_pdf(sys.argv[1], sys.argv[2])
