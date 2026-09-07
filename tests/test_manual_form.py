"""Regression checks for manual metadata entered inside a Streamlit form."""

from streamlit.testing.v1 import AppTest


def test_title_can_be_entered_and_submitted_when_inference_found_no_title() -> None:
    app = AppTest.from_string(
        "from dashboard_manual_entry import render_verification_form\n"
        "import streamlit as st\n"
        "payload = render_verification_form(suggestions={}, "
        "job_description='Responsibilities: Analyze operational data with Python and SQL.')\n"
        "if payload['submitted']:\n"
        "    st.write('Submitted title: ' + payload['title'])\n"
    ).run()
    submit = next(button for button in app.button if button.label == "Save Target Job")
    assert not submit.disabled, "Form cannot submit a title that was not inferred before rendering"
    next(field for field in app.text_input if field.label == "Job title").set_value("Data Analyst")
    submit.click().run()
    assert not app.exception
    assert any("Submitted title: Data Analyst" in item.value for item in app.markdown)
