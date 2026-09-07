from streamlit.testing.v1 import AppTest


def test_demo_resume_is_readonly_without_personal_redirect():
    app = AppTest.from_string('''
import streamlit as st
from types import SimpleNamespace
from workspace import demo_workspace
from dashboard_settings import render_candidate_workspace_setup
st.session_state["workspace_mode"] = "Demo"
render_candidate_workspace_setup(demo_workspace(), SimpleNamespace(render_page_header=lambda title, subtitle: st.header(title)))
''').run()
    assert not app.exception
    assert app.session_state["workspace_mode"] == "Demo"
    assert "Demo resume ready" in app.success[0].value
    assert len(app.text_input) == 4
    assert all(field.disabled for field in app.text_input)
    assert app.text_input[0].value == "Alex Morgan"
    assert all(button.disabled for button in app.button)
    assert len(app.get("file_uploader")) == 3
    assert all(upload.proto.disabled for upload in app.get("file_uploader"))
