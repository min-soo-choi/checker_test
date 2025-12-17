# features/ko_work.py
# -*- coding: utf-8 -*-

import html
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


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
        with st.expander("🔎 미리보기", expanded=True):
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
                        st.session_state["ko_work_output_raw"] = result.output_text
                        st.session_state["ko_work_output_edited"] = result.output_text
                        st.session_state["ko_work_output_final"] = result.output_text

        # 결과 표시
        result: Optional[WorkResult] = st.session_state.get("ko_work_result")
        if not result:
            st.caption("OCR 텍스트를 넣고 ‘실행’을 누르면 결과가 나와.")
            return

        if not result.ok:
            st.error(result.error)
            return

        st.markdown(f"### ✅ {result.title}")

        edited = st.text_area(
            "결과 텍스트 (수정 가능)",
            value=st.session_state.get("ko_work_output_edited", result.output_text),
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
                st.session_state["ko_work_input"] = st.session_state.get("ko_work_output_final", edited)
                st.success("OCR 입력을 최종본으로 교체했어. 필요하면 다시 실행해봐.")
                st.rerun()

        final_text = st.session_state.get("ko_work_output_final", edited)
        st.markdown("#### 📌 최종 확정본(복사용)")
        st.code(final_text, language="text")

        if result.data:
            st.json(result.data, expanded=False)
