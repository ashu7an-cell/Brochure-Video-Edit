#!/usr/bin/env python3
"""
tools/phone_redactor.py — Streamlit UI for redact_phones.py, wrapped as a
function so it can be embedded as one "screen" inside the combined app
instead of being its own top-level Streamlit script.
"""
import streamlit as st

from redact_phones import process_pdf_bytes, DEFAULT_MAX_SIZE_BYTES


def run():
    st.title("📄 Phone Number Redactor")
    st.write(
        "Upload a PDF and this tool will detect Indian phone numbers and visually "
        "cover them with a background-colored patch. If the file is over "
        f"{DEFAULT_MAX_SIZE_BYTES / 1_000_000:.0f} MB, it will also be compressed "
        "down to that size."
    )

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], key="phone_redactor_uploader")

    if uploaded_file is not None:
        if st.button("Redact phone numbers", key="phone_redactor_button"):
            with st.spinner("Processing PDF..."):
                try:
                    result_bytes, total_found = process_pdf_bytes(uploaded_file.read())
                    size_mb = len(result_bytes) / 1_000_000

                    if total_found:
                        st.success(f"Done! Masked {total_found} phone number(s).")
                    else:
                        st.warning("No phone numbers were detected in this PDF.")

                    if len(result_bytes) > DEFAULT_MAX_SIZE_BYTES:
                        st.warning(
                            f"Final size is {size_mb:.1f} MB - couldn't compress it "
                            f"under {DEFAULT_MAX_SIZE_BYTES / 1_000_000:.0f} MB even at "
                            "the lowest quality setting."
                        )
                    else:
                        st.caption(f"Final size: {size_mb:.1f} MB")

                    st.download_button(
                        label="⬇️ Download masked PDF",
                        data=result_bytes,
                        file_name=f"masked_{uploaded_file.name}",
                        mime="application/pdf",
                        key="phone_redactor_download",
                    )
                except Exception as e:
                    st.error(f"Something went wrong while processing the PDF: {e}")
    else:
        st.info("Upload a PDF to get started.")
