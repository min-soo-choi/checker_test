# features/ko_work.py
# -*- coding: utf-8 -*-
import html
from typing import List, Dict
import re
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List


def count_anchor_matches(text: str, anchors: List[str]) -> Dict[str, int]:
    """
    원문에서 anchor(문자열)가 몇 번 등장하는지 카운트.
    - 정규식이 아니라 '문자 그대로' 매칭
    - 겹침(overlap)은 보통 필요 없어서 기본 count로 충분
    """
    if not text:
        return {a: 0 for a in anchors if a.strip()}

    t = text.replace("\r\n", "\n")
    counts: Dict[str, int] = {}

    for a in anchors:
        a = a.strip()
        if not a:
            continue
        # re.escape로 안전하게 literal 매칭
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
    # 화면에 그대로 보이도록 escape
    escaped = html.escape(t)

    # anchors 없으면 그냥 출력
    if not anchors:
        return f"<pre style='white-space: pre-wrap; margin:0;'>{escaped}</pre>"

    # 길이 긴 anchor부터 처리(짧은게 긴걸 덮어쓰는 문제 방지)
    anchors_sorted = sorted([a for a in anchors if a.strip()], key=len, reverse=True)

    # anchor는 사용자 입력이므로 정규식 안전 처리
    for a in anchors_sorted:
        pat = re.compile(re.escape(a))
        # 하이라이트 + 줄바꿈 마커
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


def break_after_anchors(text: str, anchors: List[str]) -> str:
    """
    anchors(문자열) 뒤에서 줄바꿈을 삽입.
    사용자는 정규식을 몰라도 되도록 re.escape 처리.
    - 이미 줄바꿈이 있는 경우에도 '줄 단위 정리'는 수행 (anchor 추가 삽입은 그대로 적용)
    """
    if not text:
        return ""

    t = _normalize_ocr_text(text)
    if not anchors:
        # 줄 정리만
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        return "\n".join(lines)

    # anchors 뒤 공백(또는 줄끝)을 줄바꿈으로
    for a in anchors:
        escaped = re.escape(a)
        # anchor 뒤에 공백/탭/줄끝이 있을 때 줄바꿈으로 정리
        # - OCR이 한 줄로 붙은 경우: 공백을 \n 로 변환
        # - 이미 줄바꿈이 있는 경우: 영향 최소 (뒤 공백 정리 수준)
        t = re.sub(rf"({escaped})[ \t]*", r"\1\n", t)

    # 후처리: 빈 줄 제거 + 각 줄 trim
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    return "\n".join(lines)


def format_poetry(text: str, poetry_anchors: List[str]) -> str:
    """
    운문: 줄바꿈만 (들여쓰기 없음)
    """
    return break_after_anchors(text, poetry_anchors)


def format_prose(text: str, prose_anchors: List[str], indent: str = " ") -> str:
    """
    산문: anchors로 문단 줄바꿈 + 각 문단 첫머리 공백 1칸
    """
    broken = break_after_anchors(text, prose_anchors)
    lines = [ln.strip() for ln in broken.split("\n") if ln.strip()]
    return "\n".join(indent + ln for ln in lines)

def has_valid_anchors(anchors: list[str]) -> bool:
    return bool(anchors and any(a.strip() for a in anchors))

# =========================
# Action: 작품 들여쓰기
# =========================

@register_action("작품 들여쓰기")
def action_indent_work(text: str, params: Dict[str, Any]) -> WorkResult:
    mode = (params.get("mode") or "산문").strip()
    poetry_anchors = params.get("poetry_anchors") or []
    prose_anchors = params.get("prose_anchors") or []
    indent = params.get("indent", " ")

    if mode == "운문":
        output = format_poetry(text, poetry_anchors)
        return WorkResult(
            ok=True,
            title="운문 줄바꿈 결과",
            output_text=output,
            data={"mode": "운문", "anchors_used": poetry_anchors},
        )

    # default: 산문
    output = format_prose(text, prose_anchors, indent=indent)
    return WorkResult(
        ok=True,
        title="산문 문단 줄바꿈 + 들여쓰기 결과",
        output_text=output,
        data={"mode": "산문", "anchors_used": prose_anchors, "indent_len": len(indent)},
    )


# =========================
# Streamlit Tab Renderer
# =========================

def render_ko_work_tab(tab, st, *, review_korean_text=None):
    with tab:
        st.subheader("🧰 국어 작업")

        text = st.text_area("OCR 텍스트 입력", height=260, key="ko_work_input")

        # 기능 선택 (향후 기능 추가 대비)
        action_key = st.selectbox("작업 선택", options=list(ACTIONS.keys()), key="ko_work_action")

        # 모드 선택
        mode = st.radio("형태 선택", ["운문", "산문"], horizontal=True, key="ko_work_mode")

        # anchors 입력
        if mode == "운문":
            anchors_text = st.text_area(
                "운문 줄바꿈 기준(한 줄에 하나씩) — 해당 구절 뒤에서 줄바꿈",
                value="웃지 마라\n검을소냐\n하노라",
                height=110,
                key="ko_work_poetry_anchors",
            )
            poetry_anchors = _split_anchors(anchors_text)
            prose_anchors = []
        else:
            anchors_text = st.text_area(
                "산문 문단 구분 기준(한 줄에 하나씩) — 해당 문장/구절 뒤에서 줄바꿈",
                value="되었다.\n들었다.",
                height=110,
                key="ko_work_prose_anchors",
            )
            prose_anchors = _split_anchors(anchors_text)
            poetry_anchors = []
            
        active_anchors = poetry_anchors if mode == "운문" else prose_anchors
        anchors_ok = has_valid_anchors(active_anchors)

        match_counts = count_anchor_matches(text, active_anchors) if text.strip() and anchors_ok else {}
        missing_anchors = [a for a, c in match_counts.items() if c == 0]

        if text.strip() and anchors_ok and missing_anchors:
            st.warning(
                "⚠️ 입력한 anchors 중 원문에 존재하지 않는 항목이 있어요:\n\n- "
                + "\n- ".join(missing_anchors[:10])
                + (f"\n\n(외 {len(missing_anchors)-10}개)" if len(missing_anchors) > 10 else "")
            )


        # ✅ 현재 모드에 따라 anchors 선택 (여기서 확정)
        active_anchors = poetry_anchors if mode == "운문" else prose_anchors

        # ✅ anchors 없을 때 경고(실행 전)
        anchors_ok = has_valid_anchors(active_anchors)
        if not anchors_ok:
            st.warning(
                "⚠️ 줄바꿈 기준(anchors)이 입력되지 않았습니다.\n\n"
                "- 한 줄에 하나씩 입력해 주세요.\n"
                "- 입력한 구절 **뒤에서 줄바꿈**이 적용됩니다."
            )

        # --- 미리보기 ---
        with st.expander("🔎 anchors 적용 미리보기(원문 하이라이트)", expanded=True):
            if not text.strip():
                st.info("OCR 텍스트를 입력하면 미리보기가 표시됩니다.")
            elif not anchors_ok:
                st.info("줄바꿈 기준(anchors)을 입력하면 미리보기가 표시됩니다.")
            else:
                st.markdown("**원문에서 anchors 하이라이트 (⏎ = 줄바꿈 예상 위치)**")
                html_preview = preview_highlight_breakpoints(text, active_anchors)
                st.markdown(
                    f"<div style='background:#f7f7f7; border:1px solid #e5e5e5; border-radius:8px; padding:12px;'>{html_preview}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("**적용 결과 미리보기**")
                if mode == "운문":
                    preview_out = format_poetry(text, active_anchors)
                else:
                    preview_out = format_prose(text, active_anchors, indent=" ")
                st.code(preview_out, language="text")

        # 산문 들여쓰기(현재 요구: 공백 1칸 고정)
        indent = " "

        # params 구성
        params: Dict[str, Any] = {
            "mode": mode,
            "poetry_anchors": poetry_anchors,
            "prose_anchors": prose_anchors,
            "indent": indent,
        }

        c1, c2 = st.columns(2)
        with c1:
            run = st.button("실행", type="primary", key="ko_work_run")
        with c2:
            reset = st.button("결과 초기화", key="ko_work_reset")

        if reset:
            st.session_state.pop("ko_work_result", None)
            st.session_state.pop("ko_work_error", None)
            st.rerun()

        if run:
            st.session_state.pop("ko_work_error", None)

            if not text.strip():
                st.warning("텍스트를 입력해줘.")
            elif not anchors_ok:
                st.error("줄바꿈 기준(anchors)을 최소 1개 이상 입력해야 실행할 수 있습니다.")
            else:
                with st.spinner("처리 중..."):
                    result = run_action(action_key, text, params)
                st.session_state["ko_work_result"] = result
                
            if result and result.ok:
                st.session_state["ko_work_output_raw"] = result.output_text
                # 새 실행이면 편집본을 raw로 리셋
                st.session_state["ko_work_output_edited"] = result.output_text
                # 최종 확정본도 일단 raw로 맞춰두거나, 유지하고 싶으면 이 줄은 빼도 됨
                st.session_state["ko_work_output_final"] = result.output_text

        # 결과 표시
        result: WorkResult | None = st.session_state.get("ko_work_result")
        if not result:
            st.caption("OCR 텍스트를 넣고 ‘실행’을 누르면 결과가 나와.")
            return

        if not result.ok:
            st.error(result.error)
            return

        st.markdown(f"### ✅ {result.title}")

        # ✅ 편집 가능한 결과 텍스트
        edited = st.text_area(
            "결과 텍스트 (수정 가능)",
            value=st.session_state.get("ko_work_output_edited", result.output_text),
            height=260,
            key="ko_work_output_editor",
        )

        # text_area의 값은 key로 관리되므로, 세션에도 동기화해두면 안전
        st.session_state["ko_work_output_edited"] = edited

        c_save, c_reset_edit, c_use = st.columns(3)

        with c_save:
            if st.button("수정내용 저장(최종 확정)", type="primary", key="ko_work_save_final"):
                st.session_state["ko_work_output_final"] = edited
                st.success("최종 텍스트로 저장했어.")

        with c_reset_edit:
            if st.button("편집 취소(자동 결과로 되돌리기)", key="ko_work_reset_edit"):
                st.session_state["ko_work_output_edited"] = st.session_state.get("ko_work_output_raw", result.output_text)
                # text_area 즉시 반영 위해 rerun
                st.rerun()

        with c_use:
            # 최종본을 OCR 입력으로 다시 넣고 싶을 때(다음 단계 반복 작업용)
            if st.button("최종본을 OCR 입력으로 덮어쓰기", key="ko_work_apply_final_to_input"):
                st.session_state["ko_work_input"] = st.session_state.get("ko_work_output_final", edited)
                st.success("OCR 입력을 최종본으로 교체했어. 필요하면 다시 실행해봐.")
                st.rerun()

        # 최종 확정본 표시 (복사 확인용)
        final_text = st.session_state.get("ko_work_output_final", edited)
        st.markdown("#### 📌 최종 확정본(복사용)")
        st.code(final_text, language="text")

        if result.data:
            st.json(result.data, expanded=False)


