# features/ko_work.py
# -*- coding: utf-8 -*-

import html
import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import gspread
import streamlit as st


# =========================
# Anchor utilities
# =========================

def count_anchor_matches(text: str, anchors: List[str]) -> Dict[str, int]:
    """
    원문에서 anchor(문자열)가 몇 번 등장하는지 카운트.
    - 정규식이 아니라 '문자 그대로' 매칭
    """
    if not text:
        return {a: 0 for a in anchors if a.strip()}

    t = text.replace("\r\n", "\n")
    counts: Dict[str, int] = {}

    for a in anchors:
        a = a.strip()
        if not a:
            continue
        pat = re.compile(re.escape(a))
        counts[a] = len(pat.findall(t))

    return counts


def preview_highlight_breakpoints(text: str, anchors: List[str]) -> str:
    """
    원문에서 anchors를 하이라이트하고,
    anchors 바로 뒤에 줄바꿈 마커(⏎)를 표시하는 HTML을 반환.
    """
    if not text:
        return ""

    t = text.replace("\r\n", "\n")
    escaped = html.escape(t)

    if not anchors:
        return f"<pre style='white-space: pre-wrap; margin:0;'>{escaped}</pre>"

    anchors_sorted = sorted([a for a in anchors if a.strip()], key=len, reverse=True)

    for a in anchors_sorted:
        pat = re.compile(re.escape(a))
        escaped = pat.sub(
            lambda m: (
                "<mark style='background:#fff3a3; padding:0 2px; border-radius:2px;'>"
                f"{html.escape(m.group(0))}"
                "</mark>"
                "<span style='color:#d63384; font-weight:800; margin-left:2px;'>⏎</span>"
            ),
            escaped,
        )

    return f"<pre style='white-space: pre-wrap; margin:0;'>{escaped}</pre>"


# =========================
# Result / Action registry
# =========================

@dataclass
class WorkResult:
    ok: bool
    title: str
    output_text: str = ""
    data: Optional[Dict[str, Any]] = None
    error: str = ""


ActionFn = Callable[[str, Dict[str, Any]], WorkResult]
ACTIONS: Dict[str, ActionFn] = {}


def register_action(key: str):
    def deco(fn: ActionFn):
        ACTIONS[key] = fn
        return fn
    return deco


def run_action(action_key: str, text: str, params: Dict[str, Any]) -> WorkResult:
    fn = ACTIONS.get(action_key)
    if not fn:
        return WorkResult(ok=False, title="실행 실패", error=f"등록되지 않은 기능입니다: {action_key}")

    try:
        return fn(text, params)
    except Exception as e:
        return WorkResult(ok=False, title="실행 실패", error=str(e))


# =========================
# Google Sheet helpers
# =========================

SHEET_ID_DEFAULT: Optional[str] = None  # secrets.toml의 sheet_id 사용
SHEET_TABS = ["KOR_paragraph_db의_모의고사", "KOR_paragraph_db의_교과서"]
SERVICE_ACCOUNT_FILE = Path(__file__).resolve().parent.parent / "expertupdate-ec3c7ee5b4d6.json"


def _get_gspread_client() -> gspread.client.Client:
    """
    1) st.secrets["gcp_service_account"]에 JSON( dict )이 있을 경우 우선 사용
    2) 아니면 로컬 서비스 계정 파일 경로 사용
    """
    secrets_key = "gcp_service_account"
    if secrets_key in st.secrets:
        try:
            return gspread.service_account_from_dict(dict(st.secrets[secrets_key]))
        except Exception as e:
            raise RuntimeError("secrets['gcp_service_account'] 로드에 실패했습니다. 서비스 계정 JSON을 확인해주세요.") from e
    # Streamlit Cloud 환경 변수를 통해 전달된 경우도 지원
    env_key = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if env_key:
        try:
            import json
            return gspread.service_account_from_dict(json.loads(env_key))
        except Exception as e:
            raise RuntimeError("환경변수 GCP_SERVICE_ACCOUNT_JSON 로드에 실패했습니다. 서비스 계정 JSON 문자열을 확인해주세요.") from e

    if not SERVICE_ACCOUNT_FILE.exists():
        raise RuntimeError(
            "Google 시트 자격증명이 없습니다. "
            "st.secrets['gcp_service_account']에 서비스 계정 JSON을 넣거나 "
            f"프로젝트 루트에 {SERVICE_ACCOUNT_FILE.name} 파일을 배치해 주세요."
        )
    return gspread.service_account(filename=str(SERVICE_ACCOUNT_FILE))


def _get_sheet_id() -> str:
    """
    secrets에 sheet_id가 있으면 사용, 없으면 기본값 사용.
    기본값도 없으면 오류.
    """
    sid = st.secrets.get("sheet_id") if "sheet_id" in st.secrets else os.environ.get("SHEET_ID") or SHEET_ID_DEFAULT
    if not sid:
        raise RuntimeError("sheet_id가 설정되어 있지 않습니다. secrets.toml에 sheet_id를 추가해 주세요.")
    return sid


@st.cache_data(show_spinner=False)
def load_sheet_rows(tab_name: str) -> List[Dict[str, Any]]:
    """
    시트 한 탭의 모든 행을 로드합니다.
    """
    client = _get_gspread_client()
    sh = client.open_by_key(_get_sheet_id())
    ws = sh.worksheet(tab_name)
    return ws.get_all_records()


# =========================
# Core helpers
# =========================

def _normalize_ocr_text(text: str) -> str:
    """
    OCR 공통 정리:
    - CRLF -> LF
    - 연속 공백(2칸+) -> 1칸
    - 양끝 공백 제거
    """
    if not text:
        return ""
    t = text.replace("\r\n", "\n").strip()
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t


def _split_anchors(multiline: str) -> List[str]:
    """
    anchors 입력: 한 줄에 하나씩.
    빈 줄 제거.
    """
    if not multiline:
        return []
    return [line.strip() for line in multiline.splitlines() if line.strip()]


def has_valid_anchors(anchors: List[str]) -> bool:
    return bool(anchors and any(a.strip() for a in anchors))


def normalize_linebreaks(text: str) -> str:
    """
    - <br>, <br/>, <br /> -> \n
    - CRLF/CR -> LF
    """
    if not text:
        return ""
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def apply_non_literature_indentation(text: str, indent: str = " ") -> str:
    """
    문학 이외 갈래 규칙:
    - 사용자가 입력한 줄바꿈만 유지(\n, <br> 등)
    - 줄바꿈 된 후 각 줄 시작에 무조건 공백 1칸(기본 indent=" ")
    - 빈 줄은 유지
    """
    t = normalize_linebreaks(text)
    lines = t.split("\n")

    out: List[str] = []
    for line in lines:
        if line.strip() == "":
            out.append(line)
        else:
            out.append(indent + line.lstrip())
    return "\n".join(out)


def break_after_anchors(text: str, anchors: List[str]) -> str:
    """
    anchors(문자열) 뒤에서 줄바꿈을 삽입.
    사용자는 정규식을 몰라도 되도록 re.escape 처리.
    """
    if not text:
        return ""

    t = _normalize_ocr_text(text)
    if not anchors:
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        return "\n".join(lines)

    for a in anchors:
        escaped = re.escape(a)
        t = re.sub(rf"({escaped})[ \t]*", r"\1\n", t)

    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    return "\n".join(lines)


def format_poetry(text: str, poetry_anchors: List[str]) -> str:
    """
    운문: anchors로 줄바꿈만(들여쓰기 없음)
    """
    return break_after_anchors(text, poetry_anchors)


def format_prose(text: str, prose_anchors: List[str], indent: str = " ") -> str:
    """
    산문: anchors로 문단 줄바꿈 + 각 줄(문단) 첫머리 공백 1칸
    """
    broken = break_after_anchors(text, prose_anchors)
    lines = [ln.strip() for ln in broken.split("\n") if ln.strip()]
    return "\n".join(indent + ln for ln in lines)


# =========================
# Actions
# =========================

@register_action("2. PDF 작품 들여쓰기")
def action_indent_work(text: str, params: Dict[str, Any]) -> WorkResult:
    """
    PDF 기반:
    - 문학-운문: anchors 줄바꿈만
    - 문학-산문: anchors 줄바꿈 + 들여쓰기
    - 문학 이외: anchors 없이, 입력된 줄바꿈 유지 + 들여쓰기
    """
    mode = (params.get("mode") or "산문").strip()
    poetry_anchors = params.get("poetry_anchors") or []
    prose_anchors = params.get("prose_anchors") or []
    indent = params.get("indent", " ")

    if mode == "문학 이외":
        output = apply_non_literature_indentation(text, indent=indent)
        return WorkResult(
            ok=True,
            title="PDF-문학 이외 갈래 (줄바꿈 유지 + 들여쓰기)",
            output_text=output,
            data={"mode": "문학 이외", "indent_len": len(indent)},
        )

    if mode == "운문":
        output = format_poetry(text, poetry_anchors)
        return WorkResult(
            ok=True,
            title="PDF-운문 줄바꿈 결과",
            output_text=output,
            data={"mode": "운문", "anchors_used": poetry_anchors},
        )

    output = format_prose(text, prose_anchors, indent=indent)
    return WorkResult(
        ok=True,
        title="PDF-산문 문단 줄바꿈 + 들여쓰기 결과",
        output_text=output,
        data={"mode": "산문", "anchors_used": prose_anchors, "indent_len": len(indent)},
    )


@register_action("1. 시트 검색 작품 들여쓰기")
def action_indent_work_from_sheet(text: str, params: Dict[str, Any]) -> WorkResult:
    """
    시트 검색으로 가져온 작품:
    - 시: 들여쓰기 없음 (줄바꿈만 정규화)
    - 시 이외: 줄바꿈 유지 + 각 줄 시작 1칸 들여쓰기
    """
    work_type = (params.get("work_type") or "시 이외").strip()
    t = normalize_linebreaks(text)

    if work_type == "시":
        return WorkResult(
            ok=True,
            title="시트 검색-시 (들여쓰기 없음)",
            output_text=t,
            data={"work_type": work_type, "indent_applied": False},
        )

    output = apply_non_literature_indentation(t, indent=" ")
    return WorkResult(
        ok=True,
        title="시트 검색-시 이외 (줄바꿈 유지 + 들여쓰기)",
        output_text=output,
        data={"work_type": work_type, "indent_applied": True},
    )


# =========================
# Streamlit Tab Renderer
# =========================

def render_ko_work_tab(tab, st, *, review_korean_text=None):
    with tab:
        st.subheader("🧰 국어 작업")

        # 버튼에서 요청된 입력 덮어쓰기를 위젯 생성 전에 반영
        pending_input = st.session_state.pop("ko_work_apply_input_value", None)
        if pending_input is not None:
            st.session_state["ko_work_input"] = pending_input

        with st.expander("📄 시트에서 불러오기", expanded=False):
            st.caption("시트에서 작가명/작품명/지문 텍스트로 검색해 OCR 입력에 넣을 수 있어요.")
            sheet_tab = st.selectbox(
                "탭 선택",
                SHEET_TABS,
                key="ko_sheet_tab",
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                q_author = st.text_input("작가명 포함", key="ko_sheet_q_author")
            with c2:
                q_title = st.text_input("작품명 포함", key="ko_sheet_q_title")
            with c3:
                q_text = st.text_input("지문 텍스트 포함", key="ko_sheet_q_text")

            search = st.button("시트 조회", key="ko_sheet_search")
            if search:
                try:
                    rows = load_sheet_rows(sheet_tab)
                    q_author_l = q_author.strip().lower()
                    q_title_l = q_title.strip().lower()
                    q_text_l = q_text.strip().lower()

                    def _match(row: Dict[str, Any]) -> bool:
                        a = str(row.get("작가명", "")).lower()
                        t = str(row.get("작품명", "")).lower()
                        txt = str(row.get("지문 텍스트", "")).lower()
                        if q_author_l and q_author_l not in a:
                            return False
                        if q_title_l and q_title_l not in t:
                            return False
                        if q_text_l and q_text_l not in txt:
                            return False
                        return True

                    filtered = [r for r in rows if _match(r)]
                    st.session_state["ko_sheet_results"] = filtered
                    st.session_state["ko_sheet_selected_tab"] = sheet_tab
                    st.success(f"검색 완료: {len(filtered)}건")
                except Exception as e:
                    st.session_state["ko_sheet_results"] = []
                    st.warning(f"시트 조회 실패: {e}")

            results = st.session_state.get("ko_sheet_results", [])
            if results:
                def _hilite(text: str, needle: str) -> str:
                    if not needle.strip():
                        return html.escape(text)
                    pat = re.compile(re.escape(needle.strip()), flags=re.IGNORECASE)
                    return pat.sub(lambda m: f"<mark>{html.escape(m.group(0))}</mark>", html.escape(text))

                options = []
                for idx, row in enumerate(results):
                    title = str(row.get("작품명", "")).strip()
                    author = str(row.get("작가명", "")).strip()
                    snippet = str(row.get("지문 텍스트", "")).strip()[:60]
                    display_plain = f"{title} / {author} — {snippet}..."
                    display_html = (
                        f"{_hilite(title, q_title)} / "
                        f"{_hilite(author, q_author)} — "
                        f"{_hilite(snippet, q_text)}..."
                    )
                    options.append({"idx": idx, "plain": display_plain, "html": display_html})

                st.markdown(
                    """
                    <style>
                    div[role="radiogroup"] > label {
                        display: block;
                        background: #f8f9fb;
                        border: 1px solid #e3e6ec;
                        border-radius: 8px;
                        padding: 8px 10px;
                        margin-bottom: 6px;
                        transition: background 0.2s, border 0.2s;
                    }
                    div[role="radiogroup"] > label:hover {
                        background: #eef2f7;
                        border-color: #d4dae5;
                    }
                    div[role="radiogroup"] mark {
                        background: #fff3a3;
                        padding: 0 2px;
                        border-radius: 3px;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
                    <style>
                    div[role="radiogroup"] > label {
                        display: block;
                        background: #f8f9fb;
                        border: 1px solid #e3e6ec;
                        border-radius: 8px;
                        padding: 8px 10px;
                        margin-bottom: 6px;
                        transition: background 0.2s, border 0.2s;
                    }
                    div[role="radiogroup"] > label:hover {
                        background: #eef2f7;
                        border-color: #d4dae5;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                sel_idx = st.radio(
                    "검색 결과 선택",
                    options=[opt["idx"] for opt in options],
                    format_func=lambda x: {o["idx"]: o["plain"] for o in options}[x],
                    key="ko_sheet_selected_idx",
                )
                if st.button("이 지문을 입력에 불러오기", key="ko_sheet_apply"):
                    chosen = results[sel_idx]
                    st.session_state["ko_work_apply_input_value"] = str(chosen.get("지문 텍스트", "")).strip()
                    st.success("OCR 입력에 반영했어요. 잠시 후 갱신됩니다.")
                    st.rerun()

        with st.expander("OCR 텍스트 입력", expanded=True):
            text = st.text_area("OCR 텍스트 입력", height=260, key="ko_work_input")

        # 기능 선택 (시트 검색 작품 들여쓰기를 최우선으로 표시)
        preferred_order = ["1. 시트 검색 작품 들여쓰기", "2. PDF 작품 들여쓰기"]
        action_options = [k for k in preferred_order if k in ACTIONS]
        action_options += [k for k in ACTIONS.keys() if k not in action_options]
        default_index = action_options.index("1. 시트 검색 작품 들여쓰기") if "1. 시트 검색 작품 들여쓰기" in action_options else 0
        action_key = st.selectbox(
            "작업 선택",
            options=action_options,
            index=default_index,
            key="ko_work_action",
        )

        # 모드 선택 (작품 들여쓰기에서만 의미가 있음)
        if action_key == "1. 시트 검색 작품 들여쓰기":
            mode_label = "문학 이외 갈래"
            internal_mode = "문학 이외"
            is_non_literature = True
        else:
            mode_label = st.radio(
                "갈래 선택",
                ["문학-운문", "문학-산문", "문학 이외 갈래"],
                horizontal=True
            )
            internal_mode = (
                "운문" if mode_label == "문학-운문"
                else ("산문" if mode_label == "문학-산문" else "문학 이외")
            )
            is_non_literature = (mode_label == "문학 이외 갈래")

        # anchors 입력 (문학 이외는 anchors 없음)
        poetry_anchors: List[str] = []
        prose_anchors: List[str] = []

        if mode_label == "문학-운문":
            anchors_text = st.text_area(
                "운문 줄바꿈 기준(한 줄에 하나씩) — 해당 구절 뒤에서 줄바꿈",
                value="웃지 마라\n검을소냐\n하노라",
                height=110,
                key="ko_work_poetry_anchors",
            )
            poetry_anchors = _split_anchors(anchors_text)

        elif mode_label == "문학-산문":
            anchors_text = st.text_area(
                "산문 문단 구분 기준(한 줄에 하나씩) — 해당 문장/구절 뒤에서 줄바꿈",
                value="되었다.\n들었다.",
                height=110,
                key="ko_work_prose_anchors",
            )
            prose_anchors = _split_anchors(anchors_text)

        else:
            st.info("문학 이외 갈래는 anchors 없이, 입력된 줄바꿈(\\n 또는 <br>)만 유지하고 각 줄 시작에 들여쓰기 1칸을 적용합니다.")

        # 시트 검색 기능용 work_type UI
        work_type = "시 이외"
        if action_key == "1. 시트 검색 작품 들여쓰기":
            work_type = st.radio("시트 검색 작품 종류", ["시", "시 이외"], horizontal=True)

        # anchors 선택/검증 (문학 이외는 anchors 검증 스킵)
        active_anchors = poetry_anchors if mode_label == "문학-운문" else prose_anchors
        anchors_ok = True if is_non_literature else has_valid_anchors(active_anchors)

        # anchors 매칭 경고(문학 모드일 때만)
        if (not is_non_literature) and text.strip() and anchors_ok:
            match_counts = count_anchor_matches(text, active_anchors)
            missing_anchors = [a for a, c in match_counts.items() if c == 0]
            if missing_anchors:
                st.warning(
                    "⚠️ 입력한 anchors 중 원문에 존재하지 않는 항목이 있어요:\n\n- "
                    + "\n- ".join(missing_anchors[:10])
                    + (f"\n\n(외 {len(missing_anchors)-10}개)" if len(missing_anchors) > 10 else "")
                )

        # anchors 없을 때 경고(문학 모드에서만)
        if (not is_non_literature) and (not anchors_ok):
            st.warning(
                "⚠️ 줄바꿈 기준(anchors)이 입력되지 않았습니다.\n\n"
                "- 한 줄에 하나씩 입력해 주세요.\n"
                "- 입력한 구절 **뒤에서 줄바꿈**이 적용됩니다."
            )

        # --- 미리보기 ---
        with st.expander("🔎 미리보기", expanded=False):
            if not text.strip():
                st.info("OCR 텍스트를 입력하면 미리보기가 표시됩니다.")
            else:
                if is_non_literature:
                    st.markdown("**적용 결과 미리보기 (줄바꿈 유지 + 들여쓰기 1칸)**")
                    preview_out = apply_non_literature_indentation(text, indent=" ")
                    st.code(preview_out, language="text")
                else:
                    if not anchors_ok:
                        st.info("줄바꿈 기준(anchors)을 입력하면 미리보기가 표시됩니다.")
                    else:
                        st.markdown("**원문에서 anchors 하이라이트 (⏎ = 줄바꿈 예상 위치)**")
                        html_preview = preview_highlight_breakpoints(text, active_anchors)
                        st.markdown(
                            f"<div style='background:#f7f7f7; border:1px solid #e5e5e5; border-radius:8px; padding:12px;'>{html_preview}</div>",
                            unsafe_allow_html=True,
                        )

                        st.markdown("**적용 결과 미리보기**")
                        if internal_mode == "운문":
                            preview_out = format_poetry(text, active_anchors)
                        else:
                            preview_out = format_prose(text, active_anchors, indent=" ")
                        st.code(preview_out, language="text")

        # 산문/문학이외 들여쓰기(요구: 공백 1칸 고정)
        indent = " "

        # params 구성
        params: Dict[str, Any] = {
            "mode": internal_mode,
            "poetry_anchors": poetry_anchors,
            "prose_anchors": prose_anchors,
            "indent": indent,
            "work_type": work_type,
        }

        c1, c2 = st.columns(2)
        with c1:
            run = st.button("실행", type="primary", key="ko_work_run")
        with c2:
            reset = st.button("결과 초기화", key="ko_work_reset")

        if reset:
            st.session_state.pop("ko_work_result", None)
            st.session_state.pop("ko_work_error", None)
            st.session_state.pop("ko_work_output_raw", None)
            st.session_state.pop("ko_work_output_edited", None)
            st.session_state.pop("ko_work_output_final", None)
            st.session_state.pop("ko_work_output_editor", None)
            st.session_state.pop("ko_work_last_result_text", None)
            st.rerun()

        if run:
            st.session_state.pop("ko_work_error", None)

            if not text.strip():
                st.warning("텍스트를 입력해줘.")
            else:
                # 작품 들여쓰기 + 문학 모드일 때만 anchors 필수
                needs_anchors = (action_key == "2. PDF 작품 들여쓰기") and (not is_non_literature)
                if needs_anchors and (not anchors_ok):
                    st.error("줄바꿈 기준(anchors)을 최소 1개 이상 입력해야 실행할 수 있습니다.")
                else:
                    with st.spinner("처리 중..."):
                        result = run_action(action_key, text, params)
                    st.session_state["ko_work_result"] = result

                    if result and result.ok:
                        st.session_state["ko_work_should_sync_editor"] = True
                        st.session_state["ko_work_output_raw"] = result.output_text
                        st.session_state["ko_work_output_edited"] = result.output_text
                        st.session_state["ko_work_output_final"] = result.output_text
                        st.session_state.pop("ko_work_output_editor", None)
                        st.session_state["ko_work_last_result_text"] = result.output_text

        # 결과 표시
        result: Optional[WorkResult] = st.session_state.get("ko_work_result")
        if not result:
            st.caption("OCR 텍스트를 넣고 ‘실행’을 누르면 결과가 나와.")
            return

        if not result.ok:
            st.error(result.error)
            return

        # 실행 버튼을 막 누른 경우: 편집 영역을 최신 결과로 강제 동기화
        if st.session_state.pop("ko_work_should_sync_editor", False):
            st.session_state["ko_work_output_edited"] = result.output_text
            st.session_state["ko_work_output_final"] = result.output_text
            st.session_state["ko_work_output_raw"] = result.output_text
            st.session_state["ko_work_last_result_text"] = result.output_text

        st.markdown(f"### ✅ {result.title}")

        # 최종본(복사용) - 저장된 값이 없으면 최신 편집본/자동 결과를 사용
        edited_default = st.session_state.get("ko_work_output_edited", result.output_text)
        final_text = st.session_state.get("ko_work_output_final", edited_default)
        copy_payload = json.dumps(final_text)
        # 최종본 텍스트 + 복사 버튼을 components로 표시
        st.components.v1.html(
            f"""
            <div style="font-weight:600; margin:8px 0 4px 0;">📄 최종 확정본(복사용)</div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                <button id="ko_final_copy_btn" style="padding:6px 10px; border-radius:6px; border:1px solid #ddd; background:#f5f5f5; cursor:pointer;">
                    복사
                </button>
            </div>
            <pre style="white-space:pre-wrap; background:#f7f7f7; border:1px solid #e5e5e5; border-radius:8px; padding:12px; max-height:240px; overflow:auto;">{html.escape(final_text)}</pre>
            <script>
            const btn = document.getElementById("ko_final_copy_btn");
            const copyVal = async () => {{
              const val = {copy_payload};
              try {{
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                  await navigator.clipboard.writeText(val);
                }} else {{
                  const ta = document.createElement('textarea');
                  ta.value = val;
                  document.body.appendChild(ta);
                  ta.select();
                  document.execCommand('copy');
                  document.body.removeChild(ta);
                }}
                if (btn) {{
                  const old = btn.innerText;
                  btn.innerText = "복사 완료!";
                  setTimeout(()=>{{btn.innerText = old;}}, 1000);
                }}
              }} catch(e) {{
                if (btn) btn.innerText = "복사 실패";
              }}
            }};
            if (btn) {{
              btn.onclick = () => copyVal();
            }}
            </script>
            """,
            height=260,
            scrolling=True,
        )

        # 결과 텍스트 복사(components로 안정적 처리) - textarea 위에 배치
        st.components.v1.html(
            f"""
            <div style="display:flex; align-items:center; gap:8px; margin: 12px 0 6px 0;">
                <div style="font-weight:600; font-size:1.05rem;">✍️ 결과 텍스트 (수정 가능)</div>
                <button id="ko_edit_copy_btn" type="button"
                    style="padding:4px 8px; border-radius:6px; border:1px solid #ddd; background:#f5f5f5; cursor:pointer;">
                    복사
                </button>
                <span id="ko_edit_copy_msg" style="font-size:12px; color:#666;"></span>
            </div>
            <script>
            const editBtn = document.getElementById("ko_edit_copy_btn");
            const editMsg = document.getElementById("ko_edit_copy_msg");
            // textarea 값은 JS에서 직접 읽기 (rerun 대응)
            async function copyKoEdit() {{
              try {{
                const ta = Array.from(document.querySelectorAll('textarea[data-testid="stTextArea"]'))
                  .find(el => el.getAttribute("aria-label") === "");
                const val = ta ? (ta.value || "") : "";
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                  await navigator.clipboard.writeText(val);
                }} else {{
                  const tmp = document.createElement('textarea');
                  tmp.value = val;
                  document.body.appendChild(tmp);
                  tmp.select();
                  document.execCommand('copy');
                  document.body.removeChild(tmp);
                }}
                if (editMsg) {{
                  editMsg.textContent = "복사 완료!";
                  setTimeout(()=>{{editMsg.textContent = ""; }}, 1200);
                }}
              }} catch (e) {{
                if (editMsg) editMsg.textContent = "복사 실패";
                console.error(e);
              }}
            }}
            if (editBtn) {{
              editBtn.addEventListener("click", copyKoEdit);
            }}
            </script>
            """,
            height=60,
        )

        edited = st.text_area(
            "",
            value=st.session_state.get("ko_work_output_edited", edited_default),
            height=260,
            key="ko_work_output_editor",
        )
        st.session_state["ko_work_output_edited"] = edited

        c_save, c_reset_edit, c_use = st.columns(3)

        with c_save:
            if st.button("수정내용 저장(최종 확정)", type="primary", key="ko_work_save_final"):
                st.session_state["ko_work_output_final"] = edited
                st.success("최종 텍스트로 저장했어.")

        with c_reset_edit:
            if st.button("편집 취소(자동 결과로 되돌리기)", key="ko_work_reset_edit"):
                st.session_state["ko_work_output_edited"] = st.session_state.get("ko_work_output_raw", result.output_text)
                st.rerun()

        with c_use:
            if st.button("최종본을 OCR 입력으로 덮어쓰기", key="ko_work_apply_final_to_input"):
                st.session_state["ko_work_apply_input_value"] = final_text
                st.success("OCR 입력을 최종본으로 교체했어. 필요하면 다시 실행해봐.")
                st.rerun()

        # 모든 textarea에 복사 버튼 자동 부착(JS)
        st.markdown(
            """
            <script>
            const attachKoCopyButtons = () => {
              const areas = document.querySelectorAll('textarea[data-testid="stTextArea"]');
              areas.forEach((ta, idx) => {
                if (ta.dataset.copyAttached) return;
                ta.dataset.copyAttached = "1";
                const btn = document.createElement('button');
                btn.innerText = "복사";
                btn.type = "button";
                btn.style.marginTop = "6px";
                btn.style.padding = "4px 8px";
                btn.style.borderRadius = "6px";
                btn.style.border = "1px solid #ddd";
                btn.style.background = "#f5f5f5";
                btn.style.cursor = "pointer";
                btn.onclick = async () => {
                  const val = ta.value || "";
                  try {
                    if (navigator.clipboard && navigator.clipboard.writeText) {
                      await navigator.clipboard.writeText(val);
                    } else {
                      ta.focus();
                      ta.select();
                      const ok = document.execCommand('copy');
                      if (!ok) {
                        const tmp = document.createElement('textarea');
                        tmp.value = val;
                        document.body.appendChild(tmp);
                        tmp.select();
                        document.execCommand('copy');
                        document.body.removeChild(tmp);
                      }
                    }
                    const old = btn.innerText;
                    btn.innerText = "복사 완료!";
                    setTimeout(()=>{btn.innerText = old;}, 1000);
                  } catch(e) {
                    btn.innerText = "복사 실패";
                  }
                };
                ta.parentNode.appendChild(btn);
              });
            };
            window.addEventListener('load', attachKoCopyButtons);
            setTimeout(attachKoCopyButtons, 500);
            </script>
            """,
            unsafe_allow_html=True,
        )

        if result.data:
            st.json(result.data, expanded=False)
