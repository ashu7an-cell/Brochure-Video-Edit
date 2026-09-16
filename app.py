#!/usr/bin/env python3
"""
app.py — Combined toolbox entry point.

Instead of relying on Streamlit's file-based multipage navigation (which
behaves inconsistently across Streamlit versions/deployments), this single
script picks which tool to show with a plain sidebar radio button and calls
that tool's run() function. This is the most robust way to get two tools
under one Streamlit app/one URL.
"""
import streamlit as st

from tools import phone_redactor, video_audio_swapper

st.set_page_config(page_title="Toolbox", page_icon="🧰")

st.sidebar.title("🧰 Toolbox")
choice = st.sidebar.radio(
    "Choose a tool",
    ["📄 Phone Number Redactor", "🎬 Video Audio Swapper"],
)

if choice == "📄 Phone Number Redactor":
    phone_redactor.run()
else:
    video_audio_swapper.run()
