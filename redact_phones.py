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
# isolation. Landline numbers also almost always appear near a recognizable
# label ("Tel", "Ph", "Phone", "Mobile", "Contact Details", "Call for
# Enquiry", ...), so instead of trying to parse each number out
# individually, we look for a label ANYWHERE on a line and, if a run of
# digits sits close to it (see WORD_GAP below), redact the label and the
# digits together - covering the whole phone entry (label included) while
# leaving unrelated content elsewhere on the same line (an email address, a
# website, another column) untouched.
PHONE_LABEL_WORDS = (
    "ph", "tel", "telephone", "phone", "mobile", "mob", "cell", "fax",
    "contact details", "contact information", "contact no", "contact number",
    # Real-estate brochures often use a call-to-action banner instead of (or
    # alongside) a plain label - e.g. a "Call for Booking :" row with the
    # number(s) right after it, or a plain "Call For Enquiry:" banner.
    "call for booking", "call for bookings", "call now", "call us",
    "call for enquiry", "call for enquiries",
    "for booking", "for bookings", "book now",
    "booking enquiry", "booking enquiries", "for enquiry", "for enquiries",
    "call for details", "call for site visit", "for site visit",
)
_LABEL_ALTERNATION = '|'.join(
    re.escape(w) for w in sorted(PHONE_LABEL_WORDS, key=len, reverse=True)
)
# Matches a phone-related label ANYWHERE within a line's text (not just as
# a prefix), so a label sharing a physical line with other content (e.g. an
# "Email: ... | Website: ... Tel. ..." row) is still found.
_LABEL_ANYWHERE_RE = re.compile(
    rf'\b(?:{_LABEL_ALTERNATION})\b[:.\-]?', re.IGNORECASE,
)
# Looser fallback for call-to-action headings phrased in ways the fixed list
# above doesn't cover verbatim (e.g. "Call Us For Booking Now", "Call For
# Booking / Site Visit"). Still anchored on "call" plus "book"/"enquir" so
# it can't casually match an unrelated heading. Searched anywhere in the
# line, same as _LABEL_ANYWHERE_RE above.
_CALL_ACTION_ANYWHERE_RE = re.compile(
    r'\bcall\b[\w\s/&,]{0,30}\b(?:book(?:ing)?s?|enquir(?:y|ies))\b\s*[:.\-]?',
    re.IGNORECASE,
)
# A standalone heading with nothing else on the line at all (e.g. a lone
# "Ph:" or "Contact Details" line, with the real number elsewhere) - only
# redacted if the page also has an actual phone-number match somewhere on
# it, so a heading word with no associated number anywhere is left alone.
HEADING_ONLY_RE = re.compile(
    rf'^\s*(?:{_LABEL_ALTERNATION})\s*[:.\-]?\s*$', re.IGNORECASE,
)
_DIGIT_RUN_RE = re.compile(r'\d[\d\s\-]{4,}\d')  # a loose run of >=6 digits

# Maximum word-index distance allowed between a label (e.g. "Tel", "Call
# For Enquiry") and a nearby digit run for them to be treated as one phone
# entry and redacted together. Keeping this small is what stops the
# redaction from bleeding into unrelated content that happens to share the
# same physical line (an email address or website sitting a few words
# before a "Tel." entry, for example) - only the label and the number get
# covered, never the words in between other, farther-away content.
WORD_GAP = 4


def _union_rect(rects):
    return fitz.Rect(
        min(r.x0 for r in rects), min(r.y0 for r in rects),
        max(r.x1 for r in rects), max(r.y1 for r in rects),
    )


def _contiguous_runs(idxs):
    """Split a set of word indices into sorted contiguous runs, e.g.
    {2,3,4,9,10} -> [[2,3,4],[9,10]]. Used so a label+number pairing patches
    each physically-contiguous chunk of words with its own tight rectangle,
    instead of one bounding box stretching from the label all the way to
    the number - which would also cover any unrelated content (an email
    address, another column) that happens to sit visually between them."""
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
    """Build a searchable concatenation of a line's words plus, for each
    word, its (start, end, word_index) character-offset triple. Shared
    bookkeeping used by every regex-over-a-line helper below, so a
    string-level regex match can be translated back into the word
    index(es) it covers."""
    concat = ""
    offsets = []
    for i, (raw, disp, text) in enumerate(line_words):
        start = len(concat)
        concat += text
        offsets.append((start, len(concat), i))
        concat += " "
    return concat, offsets


def _char_span_to_word_span(s, e, offsets):
    """Translate a (start, end) character range in the line's concatenated
    text back into an inclusive (min_word_idx, max_word_idx) span."""
    word_idxs = [wi for (ws, we, wi) in offsets if ws < e and we > s]
    if not word_idxs:
        return None
    return (min(word_idxs), max(word_idxs))


def find_label_and_number_spans(line_words):
    """Find phone-related labels/headings anywhere in a line and, for each
    one, look for a nearby run of digits (a landline number, or a labeled
    mobile number) within WORD_GAP words. Only the label's words plus the
    adjacent digit-run's words are covered - never the rest of the line -
    so a label sharing a physical line with unrelated content (an email
    address, a website, another column) is not swept in.

    Returns:
      matches: list of (rect, description) for every label+number pair
        found on this line.
      orphan_label_spans: list of (min_idx, max_idx) word spans for labels
        that had NO nearby digit run on this line at all (candidate
        standalone headings, e.g. a lone "Ph:" whose number sits on a
        different line/position). The caller only redacts these if the
        page turns out to have an actual phone match somewhere else.
    """
    concat, offsets = _line_concat_and_offsets(line_words)

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
        # Find the nearest digit run within WORD_GAP words of this label.
        best = None
        best_gap = None
        for (dmin, dmax) in digit_spans:
            if dmax < lmin:
                gap = lmin - dmax
            elif dmin > lmax:
                gap = dmin - lmax
            else:
                gap = 0  # overlapping
            if gap <= WORD_GAP and (best_gap is None or gap < best_gap):
                best, best_gap = (dmin, dmax), gap

        if best:
            dmin, dmax = best

            # Geometric gate: a small word-index gap isn't reliable on its
            # own, since OCR (or a vector-outline/Illustrator-exported page)
            # can merge visually separate rows into one logical "line". A
            # label and a distant number can then be only a few word-indices
            # apart even though they sit far apart on the page. Require them
            # to also be physically close, or don't pair them - otherwise an
            # unrelated number (e.g. a PIN code) can pull in a label from
            # several rows away and the single resulting rectangle would
            # bridge over real content (an email address, another column)
            # sitting between them.
            label_rects = [line_words[i][0] for i in range(lmin, lmax + 1)]
            digit_rects = [line_words[i][0] for i in range(dmin, dmax + 1)]
            label_box = _union_rect(label_rects)
            digit_box = _union_rect(digit_rects)
            heights = [r.height for r in label_rects + digit_rects if r.height > 0]
            avg_h = sum(heights) / len(heights) if heights else 10
            gap_x = max(0, max(label_box.x0, digit_box.x0) - min(label_box.x1, digit_box.x1))
            gap_y = max(0, max(label_box.y0, digit_box.y0) - min(label_box.y1, digit_box.y1))
            if gap_x > 8.0 * avg_h or gap_y > 3.0 * avg_h:
                continue  # too far apart physically - don't pair

            cmin, cmax = min(lmin, dmin), max(lmax, dmax)
            # Cover the label's own words and the digit run's own words -
            # never anything else that merely sits between them by index.
            # A word strictly between the two spans is only pulled in if
            # it's pure punctuation (":", ".", "-", "|", ...) with no
            # letters or digits of its own, so a stray separator gets a
            # tidy patch but any real content in between (an email
            # address, a "Website:" label, ...) is left completely alone
            # even if it happens to land close by in word order.
            covered_idxs = set(range(lmin, lmax + 1)) | set(range(dmin, dmax + 1))
            for i in range(cmin, cmax + 1):
                if i in covered_idxs or i >= len(line_words):
                    continue
                text = line_words[i][2]
                if text and not any(ch.isalnum() for ch in text):
                    covered_idxs.add(i)

            # Patch each physically-contiguous run of covered words with its
            # own tight rectangle, rather than one bounding box spanning
            # from the label all the way to the number - a single box would
            # also sweep in any unrelated content sitting visually between
            # them (this was the bug that patched over the email address).
            for run in _contiguous_runs(covered_idxs):
                run_rects = [line_words[i][0] for i in run if i < len(line_words)]
                if run_rects:
                    matches.append((_union_rect(run_rects), "[labeled phone line]"))
            used_label_spans.add((lmin, lmax))

    orphan_label_spans = [s for s in label_spans if s not in used_label_spans]
    return matches, orphan_label_spans


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

    Returns a list of (rect, matched_text, word_span), where word_span is
    the inclusive (min_idx, max_idx) range of words the match covers - used
    by the caller to check whether a phone-related label sits nearby on the
    same line.
    """
    concat, offsets = _line_concat_and_offsets(line_words)

    matches = []
    for m in PHONE_RE.finditer(concat):
        matched = m.group(0)
        digit_count = sum(ch.isdigit() for ch in matched)
        # +91/91 + 10-digit mobile = 12 digits maximum.
        if digit_count not in (10, 12):
            continue

        span = _char_span_to_word_span(m.start(), m.end(), offsets)
        if span is None:
            continue
        word_idxs = sorted(range(span[0], span[1] + 1))

        # Do not require a tiny OCR gap. Only reject candidates where the
        # boxes are clearly separated into unrelated brochure regions.
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

        matches.append((fitz.Rect(x0, y0, x1, y1), matched, (word_idxs[0], word_idxs[-1])))
    return matches


def dedupe_by_overlap(matches):
    """Collapse matches whose boxes overlap by more than half of the
    smaller box's area (e.g. the same phone/heading line found once via the
    real text layer and again via OCR, or a labeled-line box that fully
    contains one or more smaller mobile-regex matches on that same line).

    When several matches overlap, the LARGEST box wins - a labeled entry
    like "Call For Enquiry: 9876543210" should redact the whole entry
    (label and number), not just whichever smaller number-only match
    happened to be recorded first."""
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
    pages; we only need to touch it once via any page that has it).

    Soft-mask (transparency/alpha) images are deliberately EXCLUDED here.
    page.get_images(full=True) returns every embedded image XObject,
    including soft masks - a soft mask is just another image entry in that
    list, and the *only* place it's identifiable as "belongs to image X as
    its mask" is via the `smask` field (index 1) of the image it's attached
    to. If a soft mask's own xref gets run through the same JPEG
    recompression path as a normal photo, it gets converted from a
    single-channel grayscale image into a 3-channel RGB JPEG - which is no
    longer a spec-valid /SMask. The base image that references it then
    fails to render correctly in many viewers (shows up blank/missing), and
    the resulting exceptions during that process also cause later
    recompression attempts on legitimate photos to be silently skipped
    (via the `except: continue` / `except: return None` guards below),
    which is why the output PDF was still staying oversized. Masks are
    small grayscale data to begin with, so leaving them untouched costs
    us essentially nothing on file size."""
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
            # CRITICAL: Never replace an image that owns a soft mask.
            # PyMuPDF's replace_image() replaces the base image stream but
            # does not safely preserve the base-image/SMask relationship for
            # all PDFs. The result can make the image render blank/missing.
            # This was the cause of essential brochure images disappearing.
            if xref in mask_xrefs or smask_xref:
                continue
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

    # Conservative mode: only JPEG photos are recompressed. Do not convert
    # PNG/palette/other image types because brochures may use them for logos,
    # illustrations, masks, or design elements where a format conversion can
    # change appearance or transparency.
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
            # Keep transparency (logos, watermarks) - PNG compresses less
            # than JPEG but won't corrupt the alpha channel.
            im.convert("RGBA").save(buf, format="PNG", optimize=True)
        else:
            # Preserve the source JPEG color model. Converting CMYK brochure
            # photography to RGB can visibly shift colors.
            jpeg_mode = "CMYK" if im.mode == "CMYK" else "RGB"
            im.convert(jpeg_mode).save(buf, format="JPEG", quality=quality, optimize=True)
    except Exception:
        return None
    return buf.getvalue()


def _visual_compression_ok(before_bytes, after_bytes, zoom=0.15,
                           max_mean_diff=7.5, max_changed_fraction=0.16):
    """Safety gate: reject compression that causes a large visual change.

    The comparison is between the already-redacted PDF and the compressed
    candidate, so intentional phone-number removal is not counted as a change.
    A missing/blank brochure image changes a large fraction of a page and is
    therefore rejected automatically.
    """
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
                print(
                    f"Safety check rejected compression on page {i+1}: "
                    f"mean pixel diff={mean_diff:.2f}, "
                    f"changed pixels={changed_fraction:.1%}"
                )
                return False

        return True
    except Exception as exc:
        # If we cannot verify the candidate, do not risk returning it.
        print(f"Safety check could not verify compressed PDF: {exc}")
        return False


def shrink_pdf_to_size(doc, max_size_bytes=DEFAULT_MAX_SIZE_BYTES):
    """Best-effort image compression with a strict no-content-loss guard.

    Only standalone JPEG image XObjects are eligible. Images with a soft mask
    are excluded because replacing their base image can break transparency.
    Every compression rung is built from the same post-redaction baseline,
    then visually compared with that baseline. If a candidate fails the
    integrity check, it is discarded. If no safe candidate reaches 25 MB,
    the function returns the last safe version even when it is larger.
    """
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

            # Never replace an image with a larger version.
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
        print(
            f"Accepted safe compression: quality={quality}, "
            f"max_dim={max_dim}, size={len(best_bytes)/1_000_000:.2f} MB"
        )

        if len(best_bytes) <= max_size_bytes:
            break

    if len(best_bytes) > max_size_bytes:
        print(
            f"WARNING: Could not safely compress to "
            f"{max_size_bytes/1_000_000:.0f} MB without risking content loss. "
            f"Returning the last verified intact PDF at "
            f"{len(best_bytes)/1_000_000:.2f} MB."
        )

    return best_bytes


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
            # Mobile numbers, wherever they sit on the line.
            for rect, matched_text, _span in find_phone_matches(lw):
                page_matches.append((rect, matched_text))

            # Any phone-related label (Tel, Contact Details, Call For
            # Enquiry, ...) paired with a nearby digit run on the SAME
            # line - covers the label AND the number together, and nothing
            # else on that line (so an email/website sharing the row is
            # left untouched).
            label_number_matches, orphan_label_spans = find_label_and_number_spans(lw)
            page_matches.extend(label_number_matches)

            # A label with no nearby number on this line (e.g. a standalone
            # "Ph:" or "Contact Details" heading, with the real number
            # elsewhere) is only stripped if the page turns out to have an
            # actual phone match somewhere else.
            for (lmin, lmax) in orphan_label_spans:
                idxs = range(lmin, lmax + 1)
                raw_rects = [lw[i][0] for i in idxs if i < len(lw)]
                if raw_rects:
                    heading_candidates.append((_union_rect(raw_rects), "[phone heading]"))

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
    final_mb = len(out_bytes) / 1_000_000
    if final_mb <= max_size_bytes / 1_000_000:
        size_msg = f"Final size: {final_mb:.2f} MB (within 25 MB limit)."
    else:
        size_msg = (
            f"Final size: {final_mb:.2f} MB. "
            "25 MB could not be reached safely, so no further compression "
            "was applied that could risk removing or corrupting brochure content."
        )
    print(f"\nDone. {total_found} phone number(s) visually covered. {size_msg} "
          f"Saved to {out_path}")
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
    final_mb = len(out_bytes) / 1_000_000
    if final_mb <= max_size_bytes / 1_000_000:
        size_msg = f"Final size: {final_mb:.2f} MB (within 25 MB limit)."
    else:
        size_msg = (
            f"Final size: {final_mb:.2f} MB. "
            "25 MB could not be reached safely; the intact PDF was retained "
            "instead of risking content loss."
        )
    print(f"\nDone. {total_found} phone number(s) visually covered. {size_msg}")
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
