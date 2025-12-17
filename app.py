# app.py
# -*- coding: utf-8 -*-
import json
import time
import re
import html
from collections import Counter
from typing import Dict, Any, List
from datetime import datetime, timezone
import uuid
import traceback
from features.ko_work import render_ko_work_tab
from features.en_work import render_en_work_tab


import streamlit as st
import google.generativeai as genai

import gspread
from google.oauth2.service_account import Credentials


# ==========================
# ✅ (정산/로깅) 설정
# ==========================
LOG_SHEET_ID = st.secrets.get("LOG_SHEET_ID")
LOG_WORKSHEET_NAME = st.secrets.get("LOG_WORKSHEET", "usage_log_worker")
LOGGING_ENABLED = bool(LOG_SHEET_ID)

# ✅ 헤더는 반드시 이 순서/개수로만 기록
LOG_HEADERS = [
    "timestamp_utc",
    "session_id",
    "feature",
    "model",
    "status",
    "latency_ms",
    "prompt_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
    "error",
]


# --------------------------
# 0. Gemini 설정 (키는 secrets에서만 읽기)
# --------------------------
API_KEY = st.secrets.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY가 secrets에 설정되어 있지 않습니다.")
    st.stop()

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash-001"
model = genai.GenerativeModel(MODEL_NAME)


def _get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def calc_gemini_flash_cost_usd(prompt_tokens: int, output_tokens: int) -> float:
    # Gemini 2.0 Flash (Standard) - text pricing (앱 내부 기준값)
    in_cost_per_1m = 0.10
    out_cost_per_1m = 0.40
    return (prompt_tokens / 1_000_000) * in_cost_per_1m + (output_tokens / 1_000_000) * out_cost_per_1m


@st.cache_resource
def _get_log_worksheet():
    if not LOGGING_ENABLED:
        return None

    try:
        sa_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
        creds = Credentials.from_service_account_info(
            sa_info,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ],
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(LOG_SHEET_ID)

        # 1) 워크시트 가져오기 (없으면 생성)
        try:
            ws = sh.worksheet(LOG_WORKSHEET_NAME)
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=LOG_WORKSHEET_NAME, rows=2000, cols=30)

        # 2) 시트가 완전 비어있으면 헤더 생성
        if len(ws.get_all_values()) == 0:
            ws.append_row(LOG_HEADERS)

        return ws

    except Exception as e:
        st.error(f"[LOG] worksheet init failed: {e}")
        st.code(traceback.format_exc())
        return None


# ==========================
# ✅ (정산/로깅) 핵심 유틸
# ==========================
def _extract_token_usage(response) -> dict:
    """
    SDK/버전별로 usage 메타가 없을 수도 있으니 최대한 방어적으로 추출.
    """
    usage = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    um = getattr(response, "usage_metadata", None)
    if um is None:
        return usage

    getter = (lambda k: um.get(k)) if isinstance(um, dict) else (lambda k: getattr(um, k, 0) or 0)

    usage["prompt_tokens"] = int(getter("prompt_token_count") or 0)
    usage["output_tokens"] = int(getter("candidates_token_count") or 0)
    total = getter("total_token_count")
    usage["total_tokens"] = int(total or (usage["prompt_tokens"] + usage["output_tokens"]))
    return usage


def _ensure_session_accumulator():
    if "billing" not in st.session_state:
        st.session_state["billing"] = {
            "total_calls": 0,
            "total_prompt_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "by_feature": {},
        }


def _accumulate_billing(feature: str, usage: dict, cost_usd: float):
    _ensure_session_accumulator()
    b = st.session_state["billing"]

    b["total_calls"] += 1
    b["total_prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
    b["total_output_tokens"] += int(usage.get("output_tokens", 0) or 0)
    b["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
    b["total_cost_usd"] += float(cost_usd or 0.0)

    if feature not in b["by_feature"]:
        b["by_feature"][feature] = {
            "calls": 0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }

    f = b["by_feature"][feature]
    f["calls"] += 1
    f["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
    f["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
    f["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
    f["cost_usd"] += float(cost_usd or 0.0)


def _log_event(feature: str, status: str, latency_ms: int, usage: dict, cost_usd: float, error: str = ""):
    """
    ✅ Gemini 호출 1회에 대한 로그를 Google Sheets에 기록 (헤더 고정)
    """
    ws = _get_log_worksheet()
    if ws is None:
        return

    now_utc = datetime.now(timezone.utc).isoformat()
    sid = _get_session_id()

    values = [
        now_utc,
        sid,
        feature,
        MODEL_NAME,
        status,
        int(latency_ms or 0),
        int(usage.get("prompt_tokens", 0) or 0),
        int(usage.get("output_tokens", 0) or 0),
        int(usage.get("total_tokens", 0) or 0),
        float(cost_usd or 0.0),
        (error or "")[:500],
    ]
    # 헤더(1행) 아래에서 첫 빈 행을 찾아 기록하고, 없으면 새 행을 추가한다.
    try:
        all_values = ws.get_all_values()
        target_row = None
        for idx in range(1, len(all_values)):
            row = all_values[idx]
            first_cell = row[0] if row else ""
            if not str(first_cell).strip():
                target_row = idx + 1  # 1-based row index
                break

        if target_row is None:
            target_row = len(all_values) + 1

        ws.update(f"A{target_row}", [values], value_input_option="RAW")
    except Exception:
        # 실패 시에는 기존 방식으로라도 기록
        ws.append_row(values, value_input_option="RAW")


def gemini_generate(feature: str, prompt: str, generation_config: dict):
    """
    ✅ 모든 Gemini 호출은 반드시 여기로 통일:
    - 토큰 추출
    - 비용 계산
    - 세션 누적
    - 시트 로그
    """
    t0 = time.time()
    try:
        resp = model.generate_content(prompt, generation_config=generation_config)
        latency_ms = int((time.time() - t0) * 1000)

        usage = _extract_token_usage(resp)
        cost_usd = calc_gemini_flash_cost_usd(usage["prompt_tokens"], usage["output_tokens"])

        _accumulate_billing(feature, usage, cost_usd)
        _log_event(feature, "ok", latency_ms, usage, cost_usd, "")

        return resp

    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        usage = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        _log_event(feature, "error", latency_ms, usage, 0.0, str(e))
        raise


# -------------------------------------------------
# 공통 유틸
# -------------------------------------------------
MAX_KO_CHUNK_LEN = 1000  # 한글 800~1200자 정도면 안정적


def split_korean_text_into_chunks(text: str, max_len: int = MAX_KO_CHUNK_LEN) -> List[str]:
    if not text:
        return []

    text = text.replace("\r\n", "\n")
    if len(text) <= max_len:
        return [text]

    chunks: List[str] = []
    n = len(text)
    start = 0

    while start < n:
        end = min(start + max_len, n)
        split_pos = text.rfind("\n", start + int(max_len * 0.4), end)
        if split_pos == -1 or split_pos <= start:
            split_pos = end

        chunk = text[start:split_pos].strip("\n")
        if chunk:
            chunks.append(chunk)

        start = split_pos

    return chunks


# -------------------------------------------------
# PDF 텍스트 정리용 프롬프트 + 래퍼
# -------------------------------------------------
PDF_RESTORE_SYSTEM_PROMPT = """
너는 PDF에서 복사해 붙여넣은 한국어 시험지/해설 텍스트를,
원문의 의미를 유지하면서 구조와 서식을 정리해 주는 도우미이다.
아래 규칙을 순서대로, 엄격하게 지켜라.

1. 텍스트 복원 및 정비
- 오타 및 깨진 글자 복원:
  입력된 텍스트에서 OCR 오류로 보이는 깨진 문자(예: , ᆢ)나 명백한 오타
  (예: 연공 지능 → 인공 지능)를 문맥에 맞게 올바른 한글, 한자, 문장부호로 복원한다.
- 원문 유지:
  텍스트의 내용을 임의로 창작하거나 왜곡하지 말고, 원문의 의미를 그대로 보존한다.

2. 헤더(제목) 텍스트 변경 규칙 (중요)
텍스트 내의 다음 키워드들을 찾아 지정된 표준 헤더로 변경한다.

[정답 해설]
- 정답
- 정답인 이유
- ( ) 정답인 이유
- 정답 해설
- 정답 설명
- 해설
- [ ] 해설
- 해설:
※ ‘해설’ 관련 표현은 모두 [정답 해설]로 통합

[오답 해설]
- 오답
- 오답 해설
- 오답 풀이
- ( ) 오답 해설
- ( ) 해설 (문맥상 오답 풀이일 경우)

[적절하지 않은 이유]
- ➜ 적절하지 않은 이유
※ 화살표(➜)가 있는 경우

[적절한 이유]
- ➜ 적절한 이유
※ 화살표(➜)가 없는 경우

[출제 의도]
- 출제 의도
- 출제의도
※ 괄호만 [] 형태로 변경

[중세의도]
- 중세의도
※ 괄호만 [] 형태로 변경

3. 헤더 순서 재배치 (구조 교정)
- 변환 작업을 마친 후, 만약 [오답 해설]이 [정답 해설]보다 먼저 나오는 경우
  텍스트 내용은 그대로 두고 헤더의 위치만 서로 맞바꾼다.
- 목표 순서:
  반드시 [정답 해설] → [오답 해설] 순서를 유지한다.
- 헤더 바로 아래에 오는 본문 내용들은 헤더와 함께 묶어서 이동시킨다.

4. 문장 및 서식 정리 (가독성 최적화)
- 줄바꿈 병합:
  문장의 중간이 어색하게 끊겨 있는 경우, 이를 공백으로 치환하여 자연스럽게 연결한다.
- 번호 목록 분리:
  문장 중간이나 끝에 원 문자(①, ②, ③… / ㉠, ㉡…)가 붙어 있는 경우
  반드시 줄을 바꾼 뒤 번호를 시작한다.
- 빈 줄 제거:
  불필요한 빈 줄(엔터 두 번 이상)은 제거하고,
  단일 줄바꿈(엔터 한 번)만 사용한다.

※ 가능한 한 기존 텍스트에 있던 원기호/선지 내용을 그대로 사용하되,
   줄 위치와 줄바꿈만 정리한다.

5. 최종 출력 형식
- 완성된 텍스트는 복사하기 쉽도록
  반드시 회색 코드 블록(Code Block) 안에 담아서 출력한다.
- 코드 블록 밖에는 어떤 설명도 출력하지 말고,
  오직 정리된 텍스트만 코드 블록 안에 넣어라.
- 코드 블록 언어 표시는 text로 사용해도 되고, 생략해도 된다.

6) 블록 간 공백 규칙
- [정답 해설] 블록과 그 다음 블록 사이에는 빈 줄을 정확히 1줄만 둔다.
- [오답 해설] 블록과 원기호(①, ②, ㉠…) 목록 사이에도 빈 줄을 정확히 1줄만 둔다.
- 블록 내부에서는 불필요한 연속 빈 줄을 제거하고 논리적으로 필요한 경우에만 단일 줄바꿈을 유지한다.

※ 정답/오답 분리 규칙 (매우 중요)

- 다음과 같은 패턴이 하나의 문단에 함께 나타나는 경우,
  반드시 정답과 오답을 분리하여 출력해야 한다.

  예:
  "정답 ①: ... ②③④⑤는 ..."
  "정답: ① ... ②, ③, ④, ⑤는 ..."
  "①은 ..., 나머지는 ..."

- 처리 규칙:
  1) "정답 ①" 또는 "정답: ①"이 발견되면
     → "정답 ①"을 단독 한 줄로 분리한다.

  2) 정답 번호 바로 뒤에 오는 설명 문장은
     반드시 [정답 해설] 아래에 배치한다.

  3) 같은 문단에서 다음 표현이 발견되면:
     - "②③④⑤는"
     - "②, ③, ④, ⑤는"
     - "나머지는"
     - "기타 보기는"
     이는 모두 오답 설명으로 간주한다.

  4) 오답 설명은 반드시 [오답 해설] 헤더 아래로 이동시킨다.

  5) 오답 번호는 "②, ③, ④, ⑤"처럼 쉼표로 구분된 형태로 통일한다.

"""


def normalize_inline_answer_marker(text: str) -> str:
    if not text:
        return text

    text = text.replace("\r\n", "\n")
    circled_nums = "①②③④⑤⑥⑦⑧⑨⑩"

    pattern = re.compile(
        rf"""
        (\b\d+\))
        \s*
        ([{circled_nums}])
        .*?
        (?=\[정답\s*해설\])
        """,
        re.VERBOSE | re.DOTALL,
    )

    def repl(m):
        qno = m.group(1)
        ans = m.group(2)
        return f"{qno} 정답: {ans}\n"

    return pattern.sub(repl, text)


def tighten_between_answer_blocks(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\n\s*\n(\[오답 해설\])", r"\n\1", text)
    return text


def restore_pdf_text(raw_text: str) -> str:
    """
    ✅ (정산 적용) PDF 정리도 gemini_generate로 호출
    """
    if not raw_text:
        return ""

    prompt = f"""{PDF_RESTORE_SYSTEM_PROMPT}

----------------------------------------
아래는 PDF에서 복사해온 원본 텍스트이다.
이 텍스트를 위 규칙에 따라 정리하라.
반드시 정리된 최종 텍스트만 코드 블록 안에 넣어서 출력할 것.

[원본 텍스트 시작]
{raw_text}
[원본 텍스트 끝]
"""

    response = gemini_generate(
        feature="pdf_restore",
        prompt=prompt,
        generation_config={"temperature": 0.0},
    )

    text = getattr(response, "text", "") or ""
    stripped = text.strip()

    m = re.match(r"^```[^\n]*\n(.*)\n```$", stripped, re.S)
    if m:
        inner = m.group(1)
        inner = normalize_inline_answer_marker(inner)
        inner = tighten_between_answer_blocks(inner)
        stripped = f"```text\n{inner}\n```"
    else:
        inner = tighten_between_answer_blocks(stripped)
        inner = normalize_inline_answer_marker(inner)
        stripped = f"```text\n{inner}\n```"

    return stripped


def remove_first_line_in_code_block(block: str) -> str:
    if not block:
        return block

    stripped = block.strip()
    m = re.match(r"^```[^\n]*\n(.*)\n```$", stripped, re.S)
    inner = m.group(1) if m else stripped

    lines = inner.splitlines()
    new_inner = "\n".join(lines[1:]) if lines else ""

    return f"```text\n{new_inner}\n```" if m else new_inner


# -------------------------------------------------
# 공통: 리포트 파싱/하이라이트
# -------------------------------------------------
def _parse_report_with_pattern(source_text: str, report: str, pattern: re.Pattern[str]) -> List[Dict[str, Any]]:
    if not report:
        return []

    lines = source_text.splitlines(keepends=True)
    line_starts: List[int] = []
    offset = 0
    for ln in lines:
        line_starts.append(offset)
        offset += len(ln)

    def index_to_line_col(idx: int) -> tuple[int, int]:
        line_no = 1
        for i, start in enumerate(line_starts):
            if i + 1 < len(line_starts) and line_starts[i + 1] <= idx:
                line_no += 1
            else:
                break
        line_start_idx = line_starts[line_no - 1]
        col_no = idx - line_start_idx + 1
        return line_no, col_no

    results: List[Dict[str, Any]] = []

    for line in report.splitlines():
        s = line.strip()
        if not s:
            continue

        m = pattern.match(s)
        if not m:
            continue

        orig = m.group(1)
        fixed = m.group(2)
        msg = m.group(3)

        idx = source_text.find(orig)
        if idx == -1:
            results.append({"original": orig, "fixed": fixed, "message": msg, "line": None, "col": None})
            continue

        line_no, col_no = index_to_line_col(idx)
        results.append({"original": orig, "fixed": fixed, "message": msg, "line": line_no, "col": col_no})

    return results


def parse_korean_report_with_positions(source_text: str, report: str) -> List[Dict[str, Any]]:
    patterns = [
        re.compile(
            r"""^-\s*['"“”‘’](.+?)['"“”‘’]\s*(?:→|->)\s*['"“”‘’](.+?)['"“”‘’]\s*:\s*(.+?)\s*['"“”‘’]?$""",
            re.UNICODE,
        ),
        re.compile(
            r"""^-\s*(.+?)\s*(?:→|->)\s*(.+?)\s*:\s*(.+?)\s*['"“”‘’]?$""",
            re.UNICODE,
        ),
    ]
    for pat in patterns:
        results = _parse_report_with_pattern(source_text, report, pat)
        if results:
            return results
    return []


def parse_english_report_with_positions(source_text: str, report: str) -> List[Dict[str, Any]]:
    pattern = re.compile(
        r"""^-\s*['"“”‘’](.+?)['"“”‘’]\s*(?:→|->)\s*['"“”‘’](.+?)['"“”‘’]\s*:\s*(.+)$""",
        re.UNICODE,
    )
    return _parse_report_with_pattern(source_text, report, pattern)


def parse_report_with_positions(source_text: str, report: str) -> List[Dict[str, Any]]:
    return parse_korean_report_with_positions(source_text, report)


def build_english_raw_report_for_highlight(raw_json: dict) -> str:
    if not isinstance(raw_json, dict):
        return ""
    mode = raw_json.get("mode")
    if mode == "two_pass_single_en":
        draft = raw_json.get("initial_report_from_detector", "") or ""
        return draft.strip()
    return (raw_json.get("content_typo_report") or "").strip()


def build_korean_raw_report_for_highlight(raw_json: dict) -> str:
    if not isinstance(raw_json, dict):
        return ""

    if raw_json.get("mode") == "chunked":
        st.info("※ 텍스트가 길어 여러 블록으로 나뉘어 검사되었으며, 1차/2차 JSON은 chunk별 raw 정보로만 존재합니다.")
        lines: List[str] = []
        for chunk in raw_json.get("chunks", []):
            idx = chunk.get("index")
            raw = chunk.get("raw") or {}
            report = (raw.get("translated_typo_report") or "").strip()
            if not report:
                continue
            if idx is not None:
                lines.append(f"# [블록 {idx}]")
            lines.append(report)
        return "\n".join(lines)

    with st.expander("1차 Detector JSON (필요 시)", expanded=False):
        st.json(raw_json.get("detector_clean", {}))
    with st.expander("2차 Judge JSON (필요 시)", expanded=False):
        st.json(raw_json.get("judge_clean", {}))

    return (raw_json.get("translated_typo_report") or "").strip()


PUNCT_COLOR_MAP = {
    ".": "#fff3cd",
    "?": "#f8d7da",
    "!": "#f5c6cb",
    ",": "#d1ecf1",
    ";": "#d6d8d9",
    ":": "#d6d8d9",
    '"': "#e0f7e9",
    "“": "#e0f7e9",
    "”": "#e0f7e9",
    "'": "#fce9d9",
    "‘": "#fce9d9",
    "’": "#fce9d9",
}

PUNCT_GROUPS: dict[str, set[str]] = {
    "종결부호(.)": {"."},
    "물음표(?)": {"?"},
    "느낌표(!)": {"!"},
    "쉼표(,)": {","},
    "쌍따옴표": {'"', "“", "”"},
    "작은따옴표": {"'", "‘", "’"},
}

PUNCT_CHARS = set(PUNCT_COLOR_MAP.keys()) | set(
    [
        '"', "'", "“", "”", "‘", "’",
        "(", ")", "[", "]", "{", "}",
        "「", "」", "『", "』", "〈", "〉", "《", "》",
        "…", "·",
    ]
)


def highlight_text_with_spans(source_text: str, spans: List[Dict[str, Any]], selected_punct_chars: set[str] | None = None) -> str:
    if not source_text:
        return ""

    intervals: List[tuple[int, int]] = []
    if spans:
        for span in spans:
            orig = span.get("original")
            if not orig:
                continue
            start = source_text.find(orig)
            if start == -1:
                continue
            end = start + len(orig)
            intervals.append((start, end))

    if not intervals:
        result_parts: List[str] = []
        for ch in source_text:
            if ch in PUNCT_CHARS and (selected_punct_chars is None or ch in selected_punct_chars):
                color = PUNCT_COLOR_MAP.get(ch, "#e2e3e5")
                result_parts.append(
                    f"<span style='background-color: {color}; padding: 0 2px; font-weight: 700; font-size: 1.05em; border-radius: 2px;'>{html.escape(ch)}</span>"
                )
            else:
                result_parts.append(html.escape(ch))
        return "".join(result_parts)

    intervals.sort(key=lambda x: x[0])
    merged_intervals: List[tuple[int, int]] = []
    cur_start, cur_end = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            merged_intervals.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged_intervals.append((cur_start, cur_end))

    result_parts: List[str] = []
    idx = 0
    interval_idx = 0
    in_error = False
    cur_err_end = None

    while idx < len(source_text):
        if interval_idx < len(merged_intervals):
            start, end = merged_intervals[interval_idx]
        else:
            start, end = None, None

        if (not in_error) and (start is not None) and (idx == start):
            in_error = True
            cur_err_end = end
            result_parts.append("<mark style='background: #fff3a3; padding: 0 2px; font-weight: 700; font-size: 1.05em; border-radius: 2px;'>")

        ch = source_text[idx]

        if in_error:
            result_parts.append(html.escape(ch))
            idx += 1
            if cur_err_end is not None and idx >= cur_err_end:
                result_parts.append("</mark>")
                in_error = False
                interval_idx += 1
                cur_err_end = None
        else:
            if ch in PUNCT_CHARS and (selected_punct_chars is None or ch in selected_punct_chars):
                color = PUNCT_COLOR_MAP.get(ch, "#e2e3e5")
                result_parts.append(
                    f"<span style='background-color: {color}; padding: 0 2px; font-weight: 700; font-size: 1.05em; border-radius: 2px;'>{html.escape(ch)}</span>"
                )
            else:
                result_parts.append(html.escape(ch))
            idx += 1

    if in_error:
        result_parts.append("</mark>")

    return "".join(result_parts)


def highlight_selected_punctuation(source_text: str, selected_keys: list[str]) -> str:
    if not source_text:
        return ""

    selected_chars: set[str] = set()
    for key in selected_keys:
        selected_chars.update(PUNCT_GROUPS.get(key, set()))

    result_parts: List[str] = []
    for ch in source_text:
        if ch in selected_chars and ch in PUNCT_COLOR_MAP:
            color = PUNCT_COLOR_MAP.get(ch, "#e2e3e5")
            result_parts.append(
                f"<span style='background-color: {color}; padding: 0 3px; font-weight: 700; font-size: 1.1em; border-radius: 3px;'>{html.escape(ch)}</span>"
            )
        else:
            result_parts.append(html.escape(ch))
    return "".join(result_parts)


# -------------------------------------------------
# ✅ Gemini(JSON) 분석 호출: 이제 feature를 받도록만 수정 (정산용)
# -------------------------------------------------
def analyze_text_with_gemini(prompt: str, feature: str, max_retries: int = 5) -> dict:
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            generation_config = {
                "response_mime_type": "application/json",
                "temperature": 0.0,
            }

            response = gemini_generate(feature, prompt, generation_config=generation_config)

            raw = getattr(response, "text", None)
            if raw is None or not str(raw).strip():
                return {
                    "suspicion_score": 5,
                    "content_typo_report": "AI 응답이 비어 있습니다.",
                    "translated_typo_report": "",
                    "markdown_report": "",
                }

            obj = json.loads(raw)
            if not isinstance(obj, dict):
                return {
                    "suspicion_score": 5,
                    "content_typo_report": f"AI 응답이 dict가 아님 (type={type(obj).__name__})",
                    "translated_typo_report": "",
                    "markdown_report": "",
                }

            return obj

        except Exception as e:
            last_error = e
            wait_time = 5 * (attempt + 1)
            print(f"[Gemini] 호출 오류 (시도 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"→ {wait_time}초 후 재시도")
                time.sleep(wait_time)

    print("[Gemini] 최대 재시도 횟수 초과.")
    return {
        "suspicion_score": 5,
        "content_typo_report": f"API 호출 실패: {last_error}",
        "translated_typo_report": "",
        "markdown_report": "",
    }


# -------------------------------------------------
# (이하: 너의 기존 로직/함수들 그대로)
#   ✅ 단, 내부에서 analyze_text_with_gemini(...) 호출하는 부분만
#      feature 문자열을 넘겨주도록 최소 수정
# -------------------------------------------------

def drop_lines_not_in_source(source_text: str, report: str) -> str:
    if not report:
        return ""

    cleaned: List[str] = []
    pattern = re.compile(
        r"""^-\s*(['"])(.+?)\1\s*(?:→|->)\s*(['"])(.+?)\3\s*:\s*(.+)$""",
        re.UNICODE,
    )

    for line in report.splitlines():
        s = line.strip()
        if not s:
            continue

        m = pattern.match(s)
        if not m:
            cleaned.append(s)
            continue

        original = m.group(2)
        if original in source_text:
            cleaned.append(s)

    return "\n".join(cleaned)


def clean_self_equal_corrections(report: str) -> str:
    if not report:
        return ""

    cleaned_lines = []
    pattern = re.compile(r"^- '(.+?)' → '(.+?)':", re.UNICODE)

    for line in report.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue

        m = pattern.match(line_stripped)
        if not m:
            cleaned_lines.append(line_stripped)
            continue

        orig = m.group(1).strip()
        fixed = m.group(2).strip()
        if orig == fixed:
            continue

        cleaned_lines.append(line_stripped)

    return "\n".join(cleaned_lines)


def drop_false_period_errors(english_text: str, report: str) -> str:
    if not report:
        return ""

    stripped = (english_text or "").rstrip()
    last_char = stripped[-1] if stripped else ""

    if last_char in [".", "?", "!"]:
        bad_phrases = [
            "마침표가 없습니다",
            "마침표가 빠져",
            "마침표가 필요",
            "마침표를 찍어야",
        ]
        cleaned_lines = []
        for line in report.splitlines():
            if any(p in line for p in bad_phrases):
                continue
            cleaned_lines.append(line.strip())
        return "\n".join(cleaned_lines)

    return report


def drop_false_korean_period_errors(report: str) -> str:
    if not report:
        return ""

    cleaned_lines = []
    pattern = re.compile(r"^- '(.+?)' → '(.+?)':", re.UNICODE)
    bad_phrases = [
        "마침표가 없습니다",
        "마침표가 빠져",
        "마침표가 필요",
        "마침표를 찍어야",
        "문장 끝에 마침표가 없",
    ]

    for line in report.splitlines():
        s = line.strip()
        if not s:
            continue

        if not any(p in s for p in bad_phrases):
            cleaned_lines.append(s)
            continue

        m = pattern.match(s)
        if not m:
            cleaned_lines.append(s)
            continue

        original = m.group(1).rstrip()
        if not original:
            cleaned_lines.append(s)
            continue

        last = original[-1]
        ok = False
        if last in ".?!":
            ok = True
        elif len(original) >= 2 and last in ['"', "'", "”", "’", "」", "』", "》", "〉", ")", "]"] and original[-2] in ".?!":
            ok = True

        if ok:
            continue
        else:
            cleaned_lines.append(s)

    return "\n".join(cleaned_lines)


def drop_false_whitespace_claims(text: str, report: str) -> str:
    if not report:
        return ""

    cleaned: list[str] = []
    pattern = re.compile(r"^- '(.+?)' → '(.+?)':.*(불필요한 공백|띄어쓰기|공백)", re.UNICODE)

    for line in report.splitlines():
        s = line.strip()
        if not s:
            continue

        m = pattern.match(s)
        if not m:
            cleaned.append(s)
            continue

        original = m.group(1)
        if not re.search(r"[ \t\u3000\u200b\u200c\u200d]", original):
            continue

        cleaned.append(s)

    return "\n".join(cleaned)


def ensure_final_punctuation_error(text: str, report: str) -> str:
    if not text or not text.strip():
        return report or ""

    s = text.rstrip()
    if not s:
        return report or ""

    last = s[-1]
    end_ok = False
    if last in ".?!":
        end_ok = True
    elif last in ['"', "'", "”", "’", "」", "』", "》", "〉", ")", "]"] and len(s) >= 2 and s[-2] in ".?!":
        end_ok = True

    if end_ok:
        return report or ""

    if report and ("마침표" in report or "문장부호" in report):
        return report

    line = "- 문단 마지막 문장 끝에 마침표(또는 물음표, 느낌표)가 빠져 있으므로 적절한 문장부호를 추가해야 합니다."
    return (report.rstrip() + "\n" + line) if report else line


def ensure_english_final_punctuation(text: str, report: str) -> str:
    if not text or not text.strip():
        return report or ""

    s = text.rstrip()
    if not s:
        return report or ""

    last = s[-1]
    end_ok = False
    if last in ".?!":
        end_ok = True
    elif last in ['"', "'", ")", "]", "”", "’"] and len(s) >= 2 and s[-2] in ".?!":
        end_ok = True

    if end_ok:
        return report or ""

    if report and ("종결부호" in report or "마침표" in report or "punctuation" in report):
        return report

    line = "- 마지막 문장이 종결부호(., ?, !)가 아닌 문장부호로 끝나 있어, 문장을 마침표 등으로 명확히 끝내는 것이 좋습니다."
    return (report.rstrip() + "\n" + line) if report else line


def ensure_sentence_end_punctuation(text: str, report: str) -> str:
    if not text or not text.strip():
        return report or ""

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    missing = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        ok = False
        if s[-1] in ".?!":
            ok = True
        elif len(s) >= 2 and s[-1] in ['"', "'", "”", "’", "」", "』", "》", "〉", ")", "]"] and s[-2] in ".?!":
            ok = True

        if not ok:
            missing.append(s)

    if not missing:
        return report or ""

    if report and any(
        key in report
        for key in ["마지막 문장에 마침표", "종결부호", "문장 끝에 마침표가 없", "마침표가 없습니다"]
    ):
        return report

    line = "- 문장 끝에 종결부호(., ?, !)가 누락된 문장이 있습니다."
    return (report.rstrip() + "\n" + line) if report else line


def dedup_korean_bullet_lines(report: str) -> str:
    if not report:
        return ""

    lines = [l.strip() for l in report.splitlines() if l.strip()]
    if not lines:
        return ""

    pattern = re.compile(r"^- '(.+?)' → '(.+?)':\s*(.+)$", re.UNICODE)

    unique_lines = []
    seen = set()
    for l in lines:
        if l not in seen:
            unique_lines.append(l)
            seen.add(l)

    entries = []
    for idx, l in enumerate(unique_lines):
        m = pattern.match(l)
        if not m:
            entries.append({"idx": idx, "raw": l, "orig": None, "msg": ""})
            continue
        orig, fixed, msg = m.group(1), m.group(2), m.group(3)
        entries.append({"idx": idx, "raw": l, "orig": orig, "msg": msg})

    to_drop = set()
    for i, e1 in enumerate(entries):
        if not e1["orig"] or "불필요한 마침표" not in e1["msg"]:
            continue
        for j, e2 in enumerate(entries):
            if i == j or not e2["orig"] or "불필요한 마침표" not in e2["msg"]:
                continue
            o1, o2 = e1["orig"], e2["orig"]
            if o1 in o2 and len(o1) < len(o2):
                to_drop.add(e1["idx"])
            elif o2 in o1 and len(o2) < len(o1):
                to_drop.add(e2["idx"])

    final_lines = [l for idx, l in enumerate(unique_lines) if idx not in to_drop]
    return "\n".join(final_lines)


def validate_and_clean_analysis(result: dict, original_english_text: str | None = None) -> dict:
    if not isinstance(result, dict):
        return {
            "suspicion_score": 5,
            "content_typo_report": "AI 응답이 유효한 JSON 형식이 아님",
            "translated_typo_report": "",
            "markdown_report": "",
        }

    score = result.get("suspicion_score")
    reports = {
        "content_typo_report": result.get("content_typo_report", "") or "",
        "translated_typo_report": result.get("translated_typo_report", "") or "",
        "markdown_report": result.get("markdown_report", "") or "",
    }

    forbidden_keywords = [
        "문맥상",
        "부적절",
        "어색",
        "더 자연스럽",
        "더 적절",
        "수정하는 것이 좋",
        "제안",
        "바꾸는 것",
        "의미를 명확히",
    ]
    for key, text in reports.items():
        if any(kw in text for kw in forbidden_keywords):
            reports[key] = ""

    forbidden_phrases = ["오류 없음", "정상", "문제 없음", "수정할 필요 없음"]
    for key, text in reports.items():
        if any(ph in text for ph in forbidden_phrases):
            reports[key] = ""

    english_report = reports["content_typo_report"]
    english_report = clean_self_equal_corrections(english_report)
    if original_english_text:
        english_report = drop_false_period_errors(original_english_text, english_report)
    reports["content_typo_report"] = english_report

    final_content = reports["content_typo_report"]
    final_translated = reports["translated_typo_report"]
    final_markdown = reports["markdown_report"]

    try:
        score = int(score)
    except Exception:
        score = 1

    score = max(1, min(5, score))

    if not final_content and not final_translated and not final_markdown:
        score = 1
    elif (final_content or final_translated or final_markdown) and score == 1:
        score = 3

    return {
        "suspicion_score": score,
        "content_typo_report": final_content,
        "translated_typo_report": final_translated,
        "markdown_report": final_markdown,
    }


# -------------------------------------------------
# 1-A. 한국어 검수 프롬프트/래퍼 (기존 그대로)
# -------------------------------------------------
def create_korean_detector_prompt_for_text(korean_text: str) -> str:
    safe_text = json.dumps(korean_text, ensure_ascii=False)
    prompt = f"""
당신은 1차 **Korean text proofreader (Detector)**입니다.
당신의 임무는 아래 한국어 텍스트에서 발생할 수 있는
**모든 잠재적 오류 후보를 최대한 많이 탐지하는 것**입니다.

이 단계에서는 약간의 과잉 탐지(False Positive)를 허용합니다.
(2차 Judge 단계에서 의미 변경·스타일 제안 등은 제거됩니다.)

출력은 반드시 아래 4개의 key만 포함하는 **단일 JSON 객체**여야 합니다.
- "suspicion_score": 1~5 정수
- "content_typo_report": "" (비워두기 — 영어용 필드)
- "translated_typo_report": "- '원문' → '수정안': 설명" 형식의 줄을 여러 개 포함한 문자열 (없으면 "")
- "markdown_report": "" (항상 빈 문자열)

모든 설명은 반드시 **한국어로** 작성해야 합니다.

------------------------------------------------------------
# 입력 텍스트 (JSON 문자열)
------------------------------------------------------------
plain_korean_json: {safe_text}

- plain_korean_json을 디코딩한 결과를 plain_korean이라고 부릅니다.
- "- '원문' → '수정안': 설명" 형식에서 '원문'은
  반드시 plain_korean 안에 실제로 존재하는 부분 문자열이어야 합니다.

------------------------------------------------------------
# 1. 이 단계에서 꼭 잡아야 하는 오류 (넓게 탐지)
------------------------------------------------------------
- 명백한 오탈자, 철자 오류
- 잘못된 띄어쓰기/붙여쓰기
- 조사·어미 오용
- 문장부호 오류 (마침표/쉼표/따옴표 짝/괄호 짝 등)
- 단어 내부가 이상하게 분리된 경우 (예: "된 다", "하 였다" 등)

이제 plain_korean_json을 디코딩하여 plain_korean을 얻은 뒤,
위 기준에 따라 "- '원문' → '수정안': 설명" 형식으로 translated_typo_report를 생성하십시오.
"""
    return prompt


def create_korean_judge_prompt_for_text(korean_text: str, draft_report: str) -> str:
    safe_text = json.dumps(korean_text, ensure_ascii=False)
    safe_report = json.dumps(draft_report, ensure_ascii=False)

    prompt = f"""
당신은 2차 **Korean text proofreader (Judge)**입니다.

역할:
- 1차 Detector가 만든 오류 후보 목록(draft_report) 중에서
  **의미를 바꾸지 않는 객관적인 오류만 남기고 나머지는 모두 제거**하는 것입니다.

------------------------------------------------------------
# 입력 1: 전체 한국어 원문 (JSON 문자열)
------------------------------------------------------------
plain_korean_json: {safe_text}

------------------------------------------------------------
# 입력 2: 1차 Detector의 후보 리포트 (JSON 문자열)
------------------------------------------------------------
draft_report_json: {safe_report}

------------------------------------------------------------
# 채택 기준 (모든 조건을 만족해야 함)
------------------------------------------------------------
1. '원문'은 plain_korean 안에 실제로 존재하는 부분 문자열이어야 한다.
2. '수정안'은 다음과 같은 **형식적·객관적 수정**만 포함해야 한다.
   - 띄어쓰기/붙여쓰기 수정
   - 조사/어미 교정
   - 명백한 오탈자·철자 오류
   - 문장부호(마침표, 쉼표, 따옴표, 괄호 등) 교정
3. 의미를 바꾸는 어휘 변경이나 문장 구조 변경은 모두 제거한다.
4. 자연스러운 표현, 문체 개선 등 **스타일 목적 수정은 모두 제거**한다.
5. plain_korean에 존재하지 않는 구절을 '원문'으로 인용한 줄은 제거한다.

------------------------------------------------------------
# 출력
------------------------------------------------------------
반환 값은 반드시 아래 4개의 key를 가진 **단일 JSON 객체**여야 합니다.
- "suspicion_score": 1~5 정수
- "content_typo_report": "" (비워두기)
- "translated_typo_report": 채택된 줄만 남긴 문자열 (없으면 "")
- "markdown_report": "" (항상 빈 문자열)
"""
    return prompt


def get_korean_stage_reports(raw_bundle: dict, final_report: str) -> dict:
    if not isinstance(raw_bundle, dict):
        raw_bundle = {}

    detector_report = ""
    judge_report = ""

    if raw_bundle.get("mode") == "chunked":
        det_lines: list[str] = []
        judge_lines: list[str] = []
        for chunk in raw_bundle.get("chunks", []):
            idx = chunk.get("index")
            raw = chunk.get("raw") or {}

            det_line = (raw.get("initial_report_from_detector") or "").strip()
            if not det_line:
                det_clean = raw.get("detector_clean") or {}
                if isinstance(det_clean, dict):
                    det_line = (det_clean.get("translated_typo_report") or "").strip()

            judge_line = (raw.get("final_report_before_rule_postprocess") or "").strip()
            if not judge_line:
                judge_clean = raw.get("judge_clean") or {}
                if isinstance(judge_clean, dict):
                    judge_line = (judge_clean.get("translated_typo_report") or "").strip()
            if not judge_line:
                judge_line = (raw.get("translated_typo_report") or "").strip()

            header = f"# [블록 {idx}]" if idx is not None else None
            if det_line:
                if header:
                    det_lines.append(header)
                det_lines.append(det_line)
            if judge_line:
                if header:
                    judge_lines.append(header)
                judge_lines.append(judge_line)

        detector_report = "\n".join(det_lines).strip()
        judge_report = "\n".join(judge_lines).strip()

    else:
        detector_clean = raw_bundle.get("detector_clean") or {}
        if isinstance(detector_clean, dict):
            detector_report = (detector_clean.get("translated_typo_report") or "").strip()

        judge_clean = raw_bundle.get("judge_clean") or {}
        if isinstance(judge_clean, dict):
            judge_report = (judge_clean.get("translated_typo_report") or "").strip()
        if not judge_report:
            judge_report = (raw_bundle.get("translated_typo_report") or "").strip()

    return {"detector": detector_report, "judge": judge_report, "final": (final_report or "").strip()}


def get_english_stage_reports(raw_bundle: dict, final_report: str) -> dict:
    if not isinstance(raw_bundle, dict):
        raw_bundle = {}

    detector_report = (raw_bundle.get("initial_report_from_detector") or "").strip()
    if not detector_report:
        detector_clean = raw_bundle.get("detector_clean") or {}
        if isinstance(detector_clean, dict):
            detector_report = (detector_clean.get("content_typo_report") or "").strip()

    judge_report = (raw_bundle.get("final_report_before_rule_postprocess") or "").strip()
    if not judge_report:
        judge_clean = raw_bundle.get("judge_clean") or {}
        if isinstance(judge_clean, dict):
            judge_report = (judge_clean.get("content_typo_report") or "").strip()
    if not judge_report:
        judge_report = (raw_bundle.get("content_typo_report") or "").strip()

    return {"detector": detector_report, "judge": judge_report, "final": (final_report or "").strip()}


def _review_korean_single_block(korean_text: str) -> Dict[str, Any]:
    detector_prompt = create_korean_detector_prompt_for_text(korean_text)
    detector_raw = analyze_text_with_gemini(detector_prompt, feature="ko_detector")
    detector_clean = validate_and_clean_analysis(detector_raw)

    draft_report = detector_clean.get("translated_typo_report", "") or ""

    judge_prompt = create_korean_judge_prompt_for_text(korean_text, draft_report)
    judge_raw = analyze_text_with_gemini(judge_prompt, feature="ko_judge")
    judge_clean = validate_and_clean_analysis(judge_raw)

    score = judge_clean.get("suspicion_score", 1)
    try:
        score = int(score)
    except Exception:
        score = 3

    final_report = judge_clean.get("translated_typo_report", "") or ""

    filtered = drop_lines_not_in_source(korean_text, final_report)
    filtered = drop_false_korean_period_errors(filtered)
    filtered = drop_false_whitespace_claims(korean_text, filtered)
    filtered = ensure_final_punctuation_error(korean_text, filtered)
    filtered = ensure_sentence_end_punctuation(korean_text, filtered)
    filtered = dedup_korean_bullet_lines(filtered)
    filtered = drop_lines_not_in_source(korean_text, filtered)

    raw_bundle = {
        "mode": "two_pass_single",
        "suspicion_score": score,
        "translated_typo_report": final_report,
        "detector_raw": detector_raw,
        "detector_clean": detector_clean,
        "judge_raw": judge_raw,
        "judge_clean": judge_clean,
        "initial_report_from_detector": draft_report,
        "final_report_before_rule_postprocess": final_report,
    }

    return {
        "score": score,
        "content_typo_report": "",
        "translated_typo_report": filtered,
        "markdown_report": "",
        "raw": raw_bundle,
    }


def review_korean_text(korean_text: str) -> Dict[str, Any]:
    chunks = split_korean_text_into_chunks(korean_text, max_len=MAX_KO_CHUNK_LEN)
    if len(chunks) == 1:
        return _review_korean_single_block(korean_text)

    merged_report_lines: List[str] = []
    raw_list: List[Dict[str, Any]] = []
    max_score = 1

    for idx, chunk in enumerate(chunks, start=1):
        res = _review_korean_single_block(chunk)
        score = res.get("score", 1) or 1
        max_score = max(max_score, score)

        report = (res.get("translated_typo_report") or "").strip()
        if report:
            merged_report_lines.append(f"# [블록 {idx}]")
            merged_report_lines.append(report)

        raw_list.append({"index": idx, "text": chunk, "raw": res.get("raw", {}), "score": score})

    merged_report = "\n".join(merged_report_lines).strip()
    if not merged_report:
        max_score = 1
    elif max_score <= 1:
        max_score = 3

    raw_bundle = {"mode": "chunked", "chunk_count": len(chunks), "chunks": raw_list, "suspicion_score": max_score}

    return {
        "score": max_score,
        "content_typo_report": "",
        "translated_typo_report": merged_report,
        "markdown_report": "",
        "raw": raw_bundle,
    }


# -------------------------------------------------
# 1-B. 영어 검수 (2-pass) - 기존 그대로, feature만 추가
# -------------------------------------------------
def create_english_detector_prompt_for_text(english_text: str) -> str:
    safe_text = json.dumps(english_text, ensure_ascii=False)
    prompt = f"""
You are the first-pass **English text proofreader (Detector)**.

Your job is to detect **as many potential objective errors as possible** in the given English text.
You may slightly over-detect (allow some false positives), because a second-pass Judge will filter them.

Your response MUST be a single JSON object with EXACTLY these keys:
- "suspicion_score": integer 1~5
- "content_typo_report": string
- "translated_typo_report": ""   (keep empty, not used here)
- "markdown_report": ""          (keep empty)

Requirements for "content_typo_report":
- It MUST be a newline-joined list of bullet lines.
- Each line MUST follow this exact format (in Korean):

  - '원문' → '수정안': 오류 설명

- All explanations MUST be written in Korean.
- '원문' MUST be an exact substring of the original English text (after decoding).

------------------------------------------------------------
# Input: English text (JSON string)
------------------------------------------------------------
plain_english_json: {safe_text}
"""
    return prompt


def create_english_judge_prompt_for_text(english_text: str, draft_report: str) -> str:
    safe_text = json.dumps(english_text, ensure_ascii=False)
    safe_report = json.dumps(draft_report, ensure_ascii=False)

    prompt = f"""
You are the second-pass **English text proofreader (Judge)**.

------------------------------------------------------------
# Input 1: original English text (JSON string)
------------------------------------------------------------
plain_english_json: {safe_text}

------------------------------------------------------------
# Input 2: Detector's candidate report (JSON string)
------------------------------------------------------------
draft_report_json: {safe_report}

------------------------------------------------------------
# Output
------------------------------------------------------------
Return EXACTLY ONE JSON object with keys:
- "suspicion_score": integer 1~5
- "content_typo_report": kept bullet lines only (or "")
- "translated_typo_report": "" 
- "markdown_report": ""
"""
    return prompt


def review_english_text(english_text: str) -> Dict[str, Any]:
    detector_prompt = create_english_detector_prompt_for_text(english_text)
    detector_raw = analyze_text_with_gemini(detector_prompt, feature="en_detector")
    detector_clean = validate_and_clean_analysis(detector_raw, original_english_text=english_text)

    draft_report = detector_clean.get("content_typo_report", "") or ""

    judge_prompt = create_english_judge_prompt_for_text(english_text, draft_report)
    judge_raw = analyze_text_with_gemini(judge_prompt, feature="en_judge")
    judge_clean = validate_and_clean_analysis(judge_raw, original_english_text=english_text)

    score = judge_clean.get("suspicion_score", 1)
    try:
        score = int(score)
    except Exception:
        score = 3
    score = max(1, min(5, score))

    final_report = judge_clean.get("content_typo_report", "") or ""

    filtered = drop_lines_not_in_source(english_text, final_report)
    filtered = ensure_english_final_punctuation(english_text, filtered)
    filtered = drop_lines_not_in_source(english_text, filtered)

    raw_bundle = {
        "mode": "two_pass_single_en",
        "suspicion_score": score,
        "content_typo_report": final_report,
        "detector_raw": detector_raw,
        "detector_clean": detector_clean,
        "judge_raw": judge_raw,
        "judge_clean": judge_clean,
        "initial_report_from_detector": draft_report,
        "final_report_before_rule_postprocess": final_report,
    }

    return {
        "score": score,
        "content_typo_report": filtered,
        "raw": raw_bundle,
    }


# -------------------------------------------------
# 공통: JSON diff / 제안 추출 (기존 유지)
# -------------------------------------------------
def summarize_json_diff(raw: dict | None, final: dict | None) -> str:
    if not isinstance(raw, dict):
        raw = {}
    if not isinstance(final, dict):
        final = {}

    lines = []
    all_keys = sorted(set(raw.keys()) | set(final.keys()))

    for key in all_keys:
        rv = raw.get(key, "<없음>")
        fv = final.get(key, "<없음>")
        if rv == fv:
            continue

        rv_str = json.dumps(rv, ensure_ascii=False) if isinstance(rv, (dict, list)) else str(rv)
        fv_str = json.dumps(fv, ensure_ascii=False) if isinstance(fv, (dict, list)) else str(fv)

        lines.append(f"- **{key}**\n  - raw: `{rv_str}`\n  - final: `{fv_str}`")

    if not lines:
        return "차이가 없습니다. (raw와 final이 동일합니다.)"

    return "\n".join(lines)


def extract_korean_suggestions_from_raw(raw: dict) -> list[str]:
    if not isinstance(raw, dict):
        return []
    collected = []
    fields = [raw.get("translated_typo_report", ""), raw.get("content_typo_report", ""), raw.get("markdown_report", "")]
    for block in fields:
        if not block:
            continue
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            if not line.startswith("- "):
                line = f"- {line}"
            collected.append(line)
    return collected


def extract_english_suggestions_from_raw(raw: dict) -> list[str]:
    if not isinstance(raw, dict):
        return []
    collected: list[str] = []
    fields = [raw.get("content_typo_report", ""), raw.get("translated_typo_report", ""), raw.get("markdown_report", "")]
    for block in fields:
        if not block:
            continue
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            if not line.startswith("- "):
                line = f"- {line}"
            collected.append(line)
    return collected


# -------------------------------------------------
# 2. Streamlit UI (기존 유지)
# -------------------------------------------------
st.set_page_config(page_title="AI 검수기 (Gemini)", page_icon="📚", layout="wide")

st.title("📚 Delta 작업자 Test (Gemini 기반)")
st.caption("한국어/영어 단일 텍스트 + 해설 양식 변환 (오탈자/형식 위주, 스타일 제안 금지).")

tab_ko, tab_en, tab_ko_work, tab_en_work, tab_pdf, tab_about, tab_debug = st.tabs(
    ["✏️ 한국어 검수", "✏️ 영어 검수", "🧰 국어 작업", "🧰 영어 작업", "📄 해설 텍스트 정리", "ℹ️ 설명", "🐞 디버그"]
)

render_ko_work_tab(
    tab_ko_work,
    st,
    review_korean_text=review_korean_text,
)

render_en_work_tab(
    tab_en_work,
    st,
    review_english_text=review_english_text,
)


# --- 한국어 검수 탭 ---
with tab_ko:
    st.subheader("한국어 텍스트 검수")
    default_ko = "이것은 테스트 문장 입니다, 그는.는 학교에 갔다,"
    text_ko = st.text_area("한국어 텍스트 입력", value=default_ko, height=220)

    if st.button("한국어 검수 실행", type="primary"):
        if not text_ko.strip():
            st.warning("먼저 한국어 텍스트를 입력해주세요.")
        else:
            with st.spinner("AI가 한국어 텍스트를 검수 중입니다..."):
                result = review_korean_text(text_ko)
            st.session_state["ko_result"] = result

    if "ko_result" in st.session_state:
        result = st.session_state["ko_result"]
        score = result.get("score", 1)
        raw_json = result.get("raw", {}) or {}
        final_report_ko = (result.get("translated_typo_report") or "").strip()
        stage_reports_ko = get_korean_stage_reports(raw_json, final_report_ko)

        final_json_display = {"의심 점수": score, "한국어 검수_report": stage_reports_ko["final"]}
        raw_json_display = {"의심 점수": raw_json.get("suspicion_score"), "한국어 검수_report": stage_reports_ko["judge"]}

        st.success("한국어 검수가 완료되었습니다!")
        st.metric("의심 점수 (1~5) 1점 -> GOOD 5점 -> BAD", f"{float(score):.2f}")

        with st.container():
            st.markdown("### 🖍 오류 위치 · 하이라이트")

            stage_choice_ko = st.radio(
                "하이라이트 기준 선택",
                ["최종(Final)", "2차 Judge", "1차 Detector"],
                horizontal=True,
                key="ko_highlight_mode",
            )

            if stage_choice_ko == "최종(Final)":
                report_for_highlight = stage_reports_ko["final"]
                mode_label = "최종(Final) 기준"
            elif stage_choice_ko == "2차 Judge":
                report_for_highlight = stage_reports_ko["judge"]
                mode_label = "2차 Judge 기준"
            else:
                report_for_highlight = stage_reports_ko["detector"]
                mode_label = "1차 Detector 기준"

            spans_ko = parse_korean_report_with_positions(text_ko, report_for_highlight)

            default_punct_keys = list(PUNCT_GROUPS.keys())
            selected_punct_keys_ko = st.multiselect(
                "문장부호 선택",
                options=default_punct_keys,
                default=default_punct_keys,
                key="ko_punct_filter",
                help="선택한 부호만 색상 표시",
            )

            st.markdown(f"#### 🔦 {mode_label} 하이라이트")
            if spans_ko:
                for span in spans_ko:
                    if span["line"] is None:
                        st.markdown(f"- `{span['original']}` → `{span['fixed']}`: {span['message']}")
                    else:
                        st.markdown(f"- L{span['line']}, C{span['col']} — `{span['original']}` → `{span['fixed']}`: {span['message']}")
            else:
                st.info(f"{mode_label}으로 하이라이트할 항목이 없습니다. 원문을 그대로 표시합니다.")

            view_mode_ko = st.radio(
                "보기 모드",
                ["오류 하이라이트", "문장부호만"],
                horizontal=True,
                key="ko_view_mode_toggle",
            )

            selected_chars_ko = set().union(*(PUNCT_GROUPS[k] for k in selected_punct_keys_ko)) if selected_punct_keys_ko else set()

            if view_mode_ko == "오류 하이라이트":
                highlighted_ko = highlight_text_with_spans(text_ko, spans_ko if spans_ko else [], selected_punct_chars=selected_chars_ko)
            else:
                highlighted_ko = highlight_selected_punctuation(text_ko, selected_punct_keys_ko)

            st.markdown(
                f"<div style='background:#f7f7f7; border:1px solid #e5e5e5; border-radius:8px; padding:12px;'>"
                f"<pre style='white-space: pre-wrap; background:transparent; margin:0; font-weight:600;'>{highlighted_ko}</pre>"
                f"</div>",
                unsafe_allow_html=True,
            )

            punct_counts_ko = Counter(ch for ch in text_ko if ch in PUNCT_COLOR_MAP)
            badge_order_ko = [(".", "종결부호"), ("?", "물음표"), ("!", "느낌표"), (",", "쉼표"), ('"', "쌍따옴표"), ("'", "작은따옴표")]
            badges_ko = []
            for ch, label in badge_order_ko:
                count = punct_counts_ko.get(ch, 0)
                color = PUNCT_COLOR_MAP.get(ch, "#e2e3e5")
                badges_ko.append(f"<span style='background-color: {color}; padding: 2px 6px; border-radius: 4px; margin-right: 6px; display: inline-block;'>{label}: {count}</span>")

            st.markdown(
                f"<div style='border: 1px solid #e9ecef; border-radius: 8px; padding: 10px; background: #f8f9fa; margin-bottom: 6px;'>{''.join(badges_ko)}</div>",
                unsafe_allow_html=True,
            )

            st.caption("※ 동일한 구절이 여러 번 등장하는 경우, 첫 번째 위치가 하이라이트될 수 있습니다.")

        with st.container():
            st.markdown("### 📊 결과 비교 · 제안")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### ✅ Final JSON (후처리 적용)")
                st.json(final_json_display, expanded=False)
            with col2:
                st.markdown("#### 🧪 Raw JSON (2차 Judge 기준)")
                st.json(raw_json_display, expanded=False)

            with st.expander("1차 Detector JSON (필요 시)", expanded=False):
                st.json(raw_json.get("detector_clean", {}))
            with st.expander("2차 Judge JSON (필요 시)", expanded=False):
                st.json(raw_json.get("judge_clean", {}))

            st.markdown("### 🛠 최종 수정 제안 사항 (최종 기준)")
            suggestions = extract_korean_suggestions_from_raw({"translated_typo_report": stage_reports_ko["final"]})
            if not suggestions:
                st.info("보고할 수정 사항이 없습니다.")
            else:
                for s in suggestions:
                    st.markdown(s)


# --- 영어 검수 탭 ---
with tab_en:
    st.subheader("영어 텍스트 검수")
    default_en = "This is a simple understaning of the Al model."
    text_en = st.text_area("English text input", value=default_en, height=220)

    if st.button("영어 검수 실행", type="primary"):
        if not text_en.strip():
            st.warning("먼저 영어 텍스트를 입력해주세요.")
        else:
            with st.spinner("AI가 영어 텍스트를 검수 중입니다..."):
                result = review_english_text(text_en)
            st.session_state["en_result"] = result

    if "en_result" in st.session_state:
        result = st.session_state["en_result"]
        score = result.get("score", 1)
        raw_json = result.get("raw", {}) or {}

        final_report_en = (result.get("content_typo_report") or "").strip()
        stage_reports_en = get_english_stage_reports(raw_json, final_report_en)

        final_json = {"의심 점수": score, "영문 검수_report": stage_reports_en["final"]}
        raw_view = {"의심 점수": raw_json.get("suspicion_score"), "영문 검수_report": stage_reports_en["judge"]}

        st.success("영어 검수가 완료되었습니다!")
        st.metric("의심 점수 (1~5) 1점 -> GOOD 5점 -> BAD", f"{float(score):.2f}")

        with st.container():
            st.markdown("### 🖍 오류 위치 · 하이라이트")

            view_mode_en = st.radio(
                "하이라이트 기준 선택",
                ["최종(Final)", "2차 Judge", "1차 Detector"],
                horizontal=True,
                key="en_highlight_mode",
            )

            if view_mode_en == "최종(Final)":
                report_for_highlight = stage_reports_en["final"]
                mode_label_en = "최종(Final) 기준"
            elif view_mode_en == "2차 Judge":
                report_for_highlight = stage_reports_en["judge"]
                mode_label_en = "2차 Judge 기준"
            else:
                report_for_highlight = stage_reports_en["detector"]
                mode_label_en = "1차 Detector 기준"

            spans_en = parse_english_report_with_positions(text_en, report_for_highlight)

            default_punct_keys = list(PUNCT_GROUPS.keys())
            selected_punct_keys_en = st.multiselect(
                "문장부호 선택",
                options=default_punct_keys,
                default=default_punct_keys,
                key="en_punct_filter",
                help="선택한 부호만 색상 표시",
            )

            st.markdown(f"#### 🔦 {mode_label_en} 하이라이트")
            if spans_en:
                for span in spans_en:
                    if span["line"] is None:
                        st.markdown(f"- `{span['original']}` → `{span['fixed']}`: {span['message']}")
                    else:
                        st.markdown(f"- L{span['line']}, C{span['col']} — `{span['original']}` → `{span['fixed']}`: {span['message']}")
            else:
                st.info(f"{mode_label_en}으로 하이라이트할 항목이 없습니다. 원문을 그대로 표시합니다.")

            selected_chars_en = set().union(*(PUNCT_GROUPS[k] for k in selected_punct_keys_en)) if selected_punct_keys_en else set()
            view_mode_en_toggle = st.radio(
                "보기 모드",
                ["오류 하이라이트", "문장부호만"],
                horizontal=True,
                key="en_view_mode_toggle",
            )

            if view_mode_en_toggle == "오류 하이라이트":
                highlighted_en = highlight_text_with_spans(text_en, spans_en if spans_en else [], selected_punct_chars=selected_chars_en)
            else:
                highlighted_en = highlight_selected_punctuation(text_en, selected_punct_keys_en)

            st.markdown(
                f"<div style='background:#f7f7f7; border:1px solid #e5e5e5; border-radius:8px; padding:12px;'>"
                f"<pre style='white-space: pre-wrap; background:transparent; margin:0; font-weight:600;'>{highlighted_en}</pre>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with st.container():
            st.markdown("### 📊 결과 비교 · 제안")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### ✅ Final JSON (후처리 적용)")
                st.json(final_json, expanded=False)
            with col2:
                st.markdown("#### 🧪 Raw JSON (2차 Judge 기준)")
                st.json(raw_view, expanded=False)

            st.markdown("#### 🔍 Raw vs Final 차이 요약")
            diff_md_en = summarize_json_diff(raw_view, final_json)
            st.markdown(diff_md_en)

            st.markdown("### 🛠 최종 수정 제안 사항 (최종 기준)")
            suggestions_en = extract_english_suggestions_from_raw({"content_typo_report": stage_reports_en["final"]})
            if not suggestions_en:
                st.info("보고할 수정 사항이 없습니다.")
            else:
                for s in suggestions_en:
                    st.markdown(s)

            with st.expander("1차 Detector JSON (필요 시)", expanded=False):
                st.json(raw_json.get("detector_clean", {}))
            with st.expander("2차 Judge JSON (필요 시)", expanded=False):
                st.json(raw_json.get("judge_clean", {}))


# --- PDF 텍스트 정리 탭 ---
with tab_pdf:
    st.subheader("📄 복사한 해설 텍스트 정리")
    st.markdown("***한 페이지 내***에 있는 텍스트만 넣어주세요")
    st.caption("PDF에서 복사한 텍스트를 붙여넣고 정리 + 첫 줄 삭제까지 할 수 있습니다.")

    pdf_raw_text = st.text_area("PDF에서 복사한 원본 텍스트", height=300, key="pdf_input_text")

    colA, colB = st.columns([1, 1])
    with colA:
        auto_trim_pdf = st.checkbox("앞뒤 공백 자동 제거", value=True, key="pdf_trim")
    with colB:
        run_pdf = st.button("텍스트 정리 실행", type="primary", key="pdf_run")

    if run_pdf:
        if not pdf_raw_text.strip():
            st.warning("먼저 텍스트를 입력해주세요.")
        else:
            text_to_send = pdf_raw_text.strip() if auto_trim_pdf else pdf_raw_text
            with st.spinner("Gemini가 텍스트를 정리하는 중입니다..."):
                cleaned_block = restore_pdf_text(text_to_send)
            st.session_state["pdf_cleaned"] = cleaned_block

    cleaned_block = st.session_state.get("pdf_cleaned")
    if cleaned_block:
        st.markdown("#### ✅ 정리된 텍스트")
        if st.button("맨 위 줄만 지우기", key="pdf_delete_first_line"):
            st.session_state["pdf_cleaned"] = remove_first_line_in_code_block(cleaned_block)
            st.rerun()
        st.markdown(st.session_state["pdf_cleaned"])


# --- 설명 탭 (원본 유지: 길이 때문에 기존과 동일하게 두어도 기능 영향 없음) ---
with tab_about:
    st.title("📘 텍스트 자동 검수기 설명서")
    st.caption("이 탭은 전체 앱의 구조와 동작 방식을 설명합니다.")
    # (원문 그대로 두면 됨 — 기능 영향 없음)


# --- 디버그 탭: ✅ 세션 누적 정산 현황만 추가 (기능 영향 없음) ---
with tab_debug:
    st.subheader("💰 세션 정산(누적)")
    b = st.session_state.get("billing")
    if not b:
        st.info("아직 Gemini 호출이 없습니다.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Calls", b["total_calls"])
        c2.metric("Total tokens", b["total_tokens"])
        c3.metric("Prompt tokens", b["total_prompt_tokens"])
        c4.metric("Cost (USD)", f"{b['total_cost_usd']:.6f}")

        st.markdown("### Feature별 누적")
        st.json(b["by_feature"], expanded=False)
