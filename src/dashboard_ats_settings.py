"""Settings controls for local employer ATS boards."""

from __future__ import annotations

from typing import Any

from company_ats import (
    ATSBoardError,
    add_ats_board,
    load_ats_boards,
    remove_ats_board,
    set_ats_board_enabled,
)


def render_ats_boards(ui: Any, *, workspace: Any, demo_mode: bool) -> None:
    """Manage per-employer public ATS boards inside the Personal workspace."""
    ui.divider()
    ui.markdown("**Company ATS · Full JD**")
    ui.caption(
        "Add an employer's Greenhouse, Lever, Ashby, or SmartRecruiters URL. "
        "Only public postings are read, and no ATS credential is required."
    )
    boards = load_ats_boards(workspace.root)
    if not boards:
        ui.info("No employer ATS boards are configured yet.")
    for index, board in enumerate(boards):
        label = f"{board.company} · {board.provider.title()}"
        left, middle, right = ui.columns([0.55, 0.25, 0.20])
        with left:
            ui.write(label)
            ui.caption(board.board_url)
        with middle:
            enabled = ui.checkbox(
                "Enabled",
                value=board.enabled,
                key=f"ats_board_enabled_{index}_{board.provider}_{board.board_token}",
                disabled=demo_mode,
            )
            if enabled != board.enabled:
                set_ats_board_enabled(workspace.root, board, enabled=enabled)
                ui.rerun()
        with right:
            if ui.button(
                "Remove",
                key=f"ats_board_remove_{index}_{board.provider}_{board.board_token}",
                disabled=demo_mode,
            ):
                remove_ats_board(workspace.root, board)
                ui.rerun()
    if demo_mode:
        ui.caption("Employer boards can only be changed in the Personal workspace.")
        return
    with ui.form("add_company_ats_board"):
        board_url = ui.text_input(
            "Employer ATS URL", placeholder="https://jobs.ashbyhq.com/company"
        )
        company = ui.text_input(
            "Company name (optional)",
            help="Used when the public ATS response does not expose a company name.",
        )
        submitted = ui.form_submit_button("Validate and add")
    if submitted:
        try:
            added = add_ats_board(workspace.root, board_url, company=company)
        except ATSBoardError as error:
            ui.error(str(error))
        else:
            ui.success(f"Added {added.company} · {added.provider.title()}.")
            ui.rerun()
