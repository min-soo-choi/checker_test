from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple
import html
import difflib
import re
import json


@dataclass
class WorkResult:
    ok: bool
    title: str
    output_text: str
    data: Optional[dict] = None
    error: str = ""


ACTIONS: Dict[str, Callable[[str, dict], WorkResult]] = {}


def register_action(name: str):
    def deco(fn):
        ACTIONS[name] = fn
        return fn
    return deco


def run_action(action_key: str, text: str, params: dict) -> WorkResult:
    fn = ACTIONS.get(action_key)
    if not fn:
        return WorkResult(ok=False, title="실행 실패", output_text="", error=f"Unknown action: {action_key}")
    try:
        result = fn(text, params)
        if result.ok and params.get("strong_brackets"):
            result.output_text = apply_strong_brackets(result.output_text)
        return result
    except Exception as e:
        return WorkResult(ok=False, title="실행 실패", output_text="", error=str(e))


# =========================================================
# 1) Diff 하이라이트 유틸 (미리보기)
# =========================================================

def _escape(s: str) -> str:
    return html.escape(s, quote=False)


def highlight_diff_html(before: str, after: str) -> Tuple[str, str]:
    """
    before/after diff를 간단하게 하이라이트.
    - after(출력)에서 바뀐/추가된 부분을 <mark>로 표시
    - before(입력)에서 바뀐 부분도 <mark>로 표시
    """
    before = before or ""
    after = after or ""

    sm = difflib.SequenceMatcher(a=before, b=after)
    out_before = []
    out_after = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a_chunk = before[i1:i2]
        b_chunk = after[j1:j2]

        if tag == "equal":
            out_before.append(_escape(a_chunk))
            out_after.append(_escape(b_chunk))
        elif tag == "delete":
            # before에서 삭제된 부분 표시
            out_before.append(f"<mark style='background:#ffd6d6;'>{_escape(a_chunk)}</mark>")
        elif tag == "insert":
            # after에서 추가된 부분 표시
            out_after.append(f"<mark style='background:#d6ffe2;'>{_escape(b_chunk)}</mark>")
        else:  # replace
            out_before.append(f"<mark style='background:#ffe9a8;'>{_escape(a_chunk)}</mark>")
            out_after.append(f"<mark style='background:#ffe9a8;'>{_escape(b_chunk)}</mark>")

    html_before = "".join(out_before)
    html_after = "".join(out_after)
    return html_before, html_after


def wrap_pre_block(inner_html: str) -> str:
    return (
        "<div style='background:#f7f7f7; border:1px solid #e5e5e5; "
        "border-radius:8px; padding:12px; white-space: pre-wrap;'>"
        "<pre style='white-space: pre-wrap; margin:0; font-weight:400;'>"
        f"{inner_html}"
        "</pre></div>"
    )


def render_strong_html(text: str) -> str:
    """
    text 안의 <strong>...</strong>만 살리고,
    나머지는 전부 escape 해서 XSS 위험을 줄임.
    """
    if not text:
        return ""

    # strong 블록을 토큰으로 잠깐 치환
    strong_blocks = []

    def _stash(m: re.Match) -> str:
        strong_blocks.append(m.group(1))
        return f"__STRONG_BLOCK_{len(strong_blocks)-1}__"

    tmp = re.sub(r"<strong>(.*?)</strong>", _stash, text, flags=re.DOTALL | re.IGNORECASE)

    # 나머지는 전부 escape
    tmp = html.escape(tmp)

    # 토큰을 strong 태그로 복원 (내용은 escape된 상태로 넣어 안전)
    for i, content in enumerate(strong_blocks):
        safe_inner = html.escape(content)
        tmp = tmp.replace(
            html.escape(f"__STRONG_BLOCK_{i}__"),
            f"<strong>{safe_inner}</strong>",
        )

    # 줄바꿈은 <br>로
    tmp = tmp.replace("\n", "<br>")

    return tmp


def render_strong_and_underline_html(text: str) -> str:
    """
    <strong>과 밑줄 span을 안전하게 렌더링.
    - 스타일이 underline을 포함하는 span/u는 <u>로 변환
    - 그 외 span은 태그 제거(내용만)
    - 나머지는 escape
    """
    if not text:
        return ""

    strong_blocks: list[str] = []
    span_blocks: list[tuple[str, bool]] = []  # (content, is_underline)

    def _stash_span(m: re.Match) -> str:
        attrs = m.group(1) or ""
        inner = m.group(2) or ""
        is_under = bool(re.search(r"underline", attrs, flags=re.IGNORECASE))
        span_blocks.append((inner, is_under))
        return f"__SPAN_BLOCK_{len(span_blocks)-1}__"

    def _stash_u(m: re.Match) -> str:
        inner = m.group(1) or ""
        span_blocks.append((inner, True))
        return f"__SPAN_BLOCK_{len(span_blocks)-1}__"

    def _stash_strong(m: re.Match) -> str:
        strong_blocks.append(m.group(1))
        return f"__STRONG_BLOCK_{len(strong_blocks)-1}__"

    tmp = text
    # underline 태그 먼저
    tmp = re.sub(r"<u>(.*?)</u>", _stash_u, tmp, flags=re.DOTALL | re.IGNORECASE)
    # span을 먼저 스태시 (underline 여부 기록)
    tmp = re.sub(r"<span(.*?)>(.*?)</span>", _stash_span, tmp, flags=re.DOTALL | re.IGNORECASE)
    # strong 스태시
    tmp = re.sub(r"<strong>(.*?)</strong>", _stash_strong, tmp, flags=re.DOTALL | re.IGNORECASE)

    tmp = html.escape(tmp)

    for i, content in enumerate(strong_blocks):
        safe_inner = html.escape(content)
        tmp = tmp.replace(
            html.escape(f"__STRONG_BLOCK_{i}__"),
            f"<strong>{safe_inner}</strong>",
        )

    for i, (content, is_under) in enumerate(span_blocks):
        safe_inner = html.escape(content)
        replacement = f"<u>{safe_inner}</u>" if is_under else safe_inner
        tmp = tmp.replace(
            html.escape(f"__SPAN_BLOCK_{i}__"),
            replacement,
        )

    tmp = tmp.replace("\n", "<br>")
    return tmp


def apply_strong_brackets(text: str) -> str:
    if not text:
        return text

    strong_blocks: list[str] = []

    def _stash(m: re.Match) -> str:
        strong_blocks.append(m.group(0))
        return f"__STRONG_BLOCK_{len(strong_blocks)-1}__"

    tmp = re.sub(r"<strong>.*?</strong>", _stash, text, flags=re.DOTALL | re.IGNORECASE)
    tmp = re.sub(r"\[([^\]]+)\]", r"<strong>[\1]</strong>", tmp)

    for i, block in enumerate(strong_blocks):
        tmp = tmp.replace(f"__STRONG_BLOCK_{i}__", block)

    return tmp


CIRCLED_0_20 = {
    0: "⓪", 1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤",
    6: "⑥", 7: "⑦", 8: "⑧", 9: "⑨", 10: "⑩", 11: "⑪",
    12: "⑫", 13: "⑬", 14: "⑭", 15: "⑮", 16: "⑯",
    17: "⑰", 18: "⑱", 19: "⑲", 20: "⑳",
}

# 원문자 숫자(①~⑨) + 원문자 알파(Ⓐ-Ⓩⓐ-ⓩ)
CIRCLED_CHAR_CLASS = r"①②③④⑤⑥⑦⑧⑨Ⓐ-Ⓩⓐ-ⓩ"

LABELS_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LABELS_LOWER = "abcdefghijklmnopqrstuvwxyz"

TEN = "__________"
MARK_L = "⟪"
MARK_R = "⟫"




def wrap_circle_numbers_clean(text: str, strong_brackets: bool = True) -> str:
    if not text or not isinstance(text, str):
        return ""

    # 1) 따옴표 정리(원문 코드 의도 반영)
    t = text.replace("”", "“").replace("”", "“")
    t = re.sub(r"[“”]", "“", t)
    t = t.replace("‘", "'").replace("’", "'")

    # 2) ( ① ) / ( Ⓐ ) -> (①)
    t = re.sub(rf"\(\s*([{CIRCLED_CHAR_CLASS}])\s*\)", r"(\1)", t)

    # 2.5) (1)~(20) -> (①)~(⑳) (0도 지원)
    def _num_to_circled(m: re.Match) -> str:
        n = int(m.group(1))
        if 0 <= n <= 20:
            return f"({CIRCLED_0_20[n]})"
        return m.group(0)

    t = re.sub(r"\(\s*([0-9]{1,2})\s*\)", _num_to_circled, t)

    # 3) 이미 (①) 형태인 것 마스킹 (중복 괄호 방지)
    #    (①) -> §CIRCLED§(①)§ 형태
    placeholder = "§CIRCLED§"
    t = re.sub(rf"\(([{CIRCLED_CHAR_CLASS}])\)", rf"{placeholder}(\1){placeholder}", t)

    # 4) 남아있는 원문자 자체를 괄호로 감싸기: ① -> (①)
    t = re.sub(rf"(?<!\()\s*([{CIRCLED_CHAR_CLASS}])\s*(?!\))", r"(\1)", t)


    # 5) 마스킹 복원 (일반 텍스트의 밑줄/언더스코어는 건드리지 않도록 고유 토큰 사용)
    t = t.replace(placeholder, "")

    # 6) 괄호 앞뒤 공백 하나로 정리
    #    " ( ① ) " 같은 걸 " (①) " 느낌으로
    t = re.sub(r"\s*\(\s*", " (", t)
    t = re.sub(r"\s*\)\s*", ") ", t)
    t = re.sub(r"[ \t]{2,}", " ", t)

    # 7) [내용] -> <strong>[내용]</strong>
    if strong_brackets:
        t = re.sub(r"\[([^\]]+)\]", r"<strong>[\1]</strong>", t)

    return t.strip()

def format_with_labels(text: str, lowercase: bool = False) -> str:
    """
    Unified version of:
      - formatWithLabels (v7)
      - formatWithLowercaseLabels

    Args:
      lowercase: False -> (A)(B)(C)...
                 True  -> (a)(b)(c)...
    """
    if not text or not isinstance(text, str):
        return ""

    labels = LABELS_LOWER if lowercase else LABELS_UPPER

    # ===== (0) 공통 정리 =====
    t = (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("\u00A0", " ")
    )
    t = re.sub(r"[\u200B\uFEFF]", "", t).strip()

    # ===== (1) (ㄱ)(ㄴ)(ㄷ)... 형식이면 그대로 유지 =====
    if re.search(r"\(ㄱ\)", t):
        parts = re.split(r"(?=\([ㄱ-ㅎ]\))", t)
        parts = [p.strip() for p in parts if p and p.strip()]
        return "    ".join(parts)

    # ===== (2) 숫자 / 원기호 / 영문 라벨 제거 → | 로 통일 =====
    t = (
        re.sub(r"(\d+\))", " | ", t)       # 1) 2) ...
        .replace("①", " | ").replace("②", " | ")
    )
    t = re.sub(r"[①-⑳]", " | ", t)        # 원기호 숫자
    t = re.sub(r"[ⓐ-ⓩ]", " | ", t)        # 원기호 알파벳
    t = re.sub(r"\([A-Za-z]\)", " | ", t) # (A) (b)
    t = re.sub(r"\|{2,}", "|", t).strip()

    # ===== (3) 줄바꿈 / 쉼표 / 슬래시 / 세미콜론 / | / [] =====
    items = re.split(r"[\n,\/;|\[\]]+", t)
    items = [it.strip() for it in items if it and it.strip()]

    if not items:
        return ""

    # ===== (4) (A)/(a) 라벨 부여 + 4칸 간격 =====
    out = []
    for i, item in enumerate(items):
        label = labels[i % 26]
        out.append(f"({label}) {item}")

    return "    ".join(out)


def add_labels_to_brackets(text: str, use_lowercase: bool = False) -> str:
    """
    Port of Apps Script addLabelsToBrackets(input, useLowercase=false)

    - normalize quotes/spaces/fullwidth brackets/punct
    - remove existing <strong>[ ... ]</strong> wrapper back to [ ... ]
    - scan () and [] with a single regex (no nesting support)
    - if inside has 2+ tokens split by comma or slash -> convert to "[ a / b ]"
    - optionally prefix with (A) / (a) labels sequentially
    - avoid duplicate label if immediately preceded by the same label
    - remove numbers right before a bracket group
    """
    if not text or not isinstance(text, str):
        return ""

    labels = LABELS_LOWER if use_lowercase else LABELS_UPPER
    label_index = 0

    src = text

    # 0) normalize: quotes/spaces/fullwidth symbols, remove strong wrapper
    src = (
        src.replace("\u00A0", " ")
           .replace("\u200B", "")
           .replace("\uFEFF", "")
    )
    # quotes
    src = re.sub(r"[“”]", '"', src)
    src = re.sub(r"[‘’]", "'", src)

    # normalize brackets (fullwidth -> ascii)
    src = re.sub(r"[［\[]", "[", src)
    src = re.sub(r"[］\]]", "]", src)
    src = re.sub(r"[（(]", "(", src)
    src = re.sub(r"[）)]", ")", src)

    # normalize punctuation
    src = re.sub(r"[，、]", ",", src)
    src = src.replace("／", "/")

    # remove <strong>[ ... ]</strong> wrapper back to [...]
    src = re.sub(r"<strong>\s*\[\s*", "[", src, flags=re.IGNORECASE)
    src = re.sub(r"\s*\]\s*</strong>", "]", src, flags=re.IGNORECASE)

    # one-pass scan for () and [] (no nesting)
    bracket_re = re.compile(r"(\[|\()([^()\[\]]*?)(\]|\))")

    # detect trailing label right before (or we already emitted)
    trailing_label_re = re.compile(r"\(\s*([A-Za-z])\s*\)\s*$")

    # new condition: remove number right before bracket
    number_before_bracket_re = re.compile(r"\d+\s*$")

    out = []
    last_index = 0

    for m in bracket_re.finditer(src):
        start, end = m.start(), m.end()
        chunk = src[last_index:start]

        # remove trailing number right before bracket
        if number_before_bracket_re.search(chunk):
            chunk = number_before_bracket_re.sub("", chunk)

        inner = m.group(2)

        # split tokens by comma or slash
        tokens = [s.strip() for s in re.split(r"[,/]", inner) if s.strip()]

        # token 1개면 그대로 (라벨 소비 X)
        if len(tokens) < 2:
            out.append(chunk + m.group(0))
            last_index = end
            continue

        cleaned = " / ".join(tokens)

        # ===== duplicate label guard =====
        label_prefix = ""
        before_text = "".join(out) + chunk
        tl = trailing_label_re.search(before_text)

        if tl:
            existing = tl.group(1)
            expected = labels[label_index] if label_index < len(labels) else ""
            if expected and existing and existing.lower() == expected.lower():
                label_index += 1
            label_prefix = ""  # don't add
        else:
            if label_index < len(labels):
                label_prefix = f"({labels[label_index]}) "
                label_index += 1
        # ===== end guard =====

        # spacing: if chunk and current out end not whitespace, insert a space
        if out:
            if out[-1] and (not out[-1][-1].isspace()) and chunk and (not chunk[0].isspace()):
                chunk = " " + chunk

        out.append(chunk + f"{label_prefix}[ {cleaned} ]")

        # handle next char spacing/punct
        next_char = src[end] if end < len(src) else ""
        if not next_char:
            out.append(" ")
        elif not re.match(r"[.,!?]", next_char):
            if not next_char.isspace():
                out.append(" ")

        last_index = end

    out.append(src[last_index:])

    # normalize spaces
    result = "".join(out)
    result = re.sub(r"[ \t\f\v]+", " ", result).strip()
    return result


def convert_commas_in_brackets(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""

    # 따옴표 정리(의미 영향 없게)
    t = re.sub(r"[“”]", '"', text)
    t = re.sub(r"[‘’]", "'", t)

    # 얕은 괄호만 (중첩 제외)
    bracket_re = re.compile(r"(\(|\[)([^()\[\]]+?)(\)|\])")

    out = []
    last_index = 0

    for m in bracket_re.finditer(t):
        open_b = m.group(1)
        inner = m.group(2)
        close_b = m.group(3)

        # 매치 전 구간 복사
        out.append(t[last_index:m.start()])

        emit = m.group(0)
        is_square = (open_b == "[" and close_b == "]")

        comma_count = inner.count(",")

        if open_b == "(" and close_b == ")" and comma_count >= 2:
            # () 내부 쉼표 2개 이상 => []로 + ' / ' 통일
            parts = [s.strip() for s in inner.split(",") if s.strip()]
            emit = f"[ {' / '.join(parts)} ]"
            is_square = True

        elif is_square:
            # [] 내부는 , 또는 / 혼용을 / 로 통일
            tokens = [s.strip() for s in re.split(r"[/,]", inner) if s.strip()]
            if len(tokens) >= 2:
                emit = f"[ {' / '.join(tokens)} ]"
            else:
                clean_inner = re.sub(r"\s+", " ", inner).strip()
                emit = f"[ {clean_inner} ]"


        if is_square:
            # 대괄호 앞 공백: 정확히 1칸(단, 줄바꿈/시작 제외)
            current = "".join(out)
            # current 끝의 스페이스만 제거(줄바꿈은 유지)
            while current.endswith(" "):
                current = current[:-1]
            out = [current]  # 재저장

            if current and not current.endswith("\n"):
                out.append(" ")

            out.append(emit)

            # 대괄호 뒤 공백: 원문에서 스페이스는 소비하고 1칸만 보장(단, 줄바꿈/끝 제외)
            next_idx = m.end()
            while next_idx < len(t) and t[next_idx] == " ":
                next_idx += 1

            next_ch = t[next_idx] if next_idx < len(t) else ""
            if next_ch and next_ch != "\n":
                out.append(" ")

            last_index = next_idx
        else:
            # []가 아닌 경우 원문 그대로
            out.append(emit)
            last_index = m.end()

    out.append(t[last_index:])

    result = "".join(out)
    # 여분 공백 수축 (전체)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result


def convert_commas_in_brackets_with_underline(
    text: str,
    *,
    add_labels: bool = False,
    use_lowercase: bool = False,
) -> str:
    """
    괄호 내부 단어 배열 정규화 + 대괄호 구간 밑줄 처리.
    - ()에서 쉼표 2개 이상이면 []로 바꾸고 ' / '로 연결 후 밑줄
    - [] 내부는 쉼표/슬래시 혼용을 ' / '로 정규화 후 밑줄
    - 괄호 앞뒤 스페이스는 1칸으로 정리(줄바꿈/문장 시작 제외)
    """
    if not isinstance(text, str) or not text:
        return ""

    # 따옴표 정리
    t = re.sub(r"[“”]", '"', text)
    t = re.sub(r"[‘’]", "'", t)

    # 불필요한 스타일(span) 제거: underline만 보존
    def _normalize_span(m: re.Match) -> str:
        attrs = m.group(1) or ""
        inner = m.group(2) or ""
        is_under = bool(re.search(r"underline", attrs, flags=re.IGNORECASE))
        return f"<u>{inner}</u>" if is_under else inner

    t = re.sub(r"<span(.*?)>(.*?)</span>", _normalize_span, t, flags=re.DOTALL | re.IGNORECASE)

    bracket_re = re.compile(r"(\(|\[)([^()\[\]]+?)(\)|\])")

    out: list[str] = []
    last_index = 0
    label_index = 0
    labels = LABELS_LOWER if use_lowercase else LABELS_UPPER

    for m in bracket_re.finditer(t):
        open_b, inner, close_b = m.group(1), m.group(2), m.group(3)

        # 매치 이전 구간 복사(바깥 쉼표 변환 없음)
        out.append(t[last_index:m.start()])

        is_square = (open_b == "[" and close_b == "]")
        comma_count = inner.count(",")

        normalized_inner = None
        if open_b == "(" and close_b == ")" and comma_count >= 2:
            tokens = [s.strip() for s in inner.split(",") if s.strip()]
            normalized_inner = " / ".join(tokens)
            is_square = True
        elif is_square:
            tokens = [s.strip() for s in re.split(r"[/,]", inner) if s.strip()]
            if len(tokens) >= 2:
                normalized_inner = " / ".join(tokens)
            else:
                normalized_inner = re.sub(r"\s+", " ", inner).strip()

        if is_square:
            # 앞 공백 정리: 연속 공백 제거 후, 줄바꿈/시작이 아니면 1칸
            current = "".join(out)
            while current.endswith(" "):
                current = current[:-1]
            out = [current]
            if current and not current.endswith("\n"):
                out.append(" ")

            emit = f'[ <span style="text-decoration: underline;">{normalized_inner}</span> ]'

            label_prefix = ""
            if add_labels:
                label_char = labels[label_index % len(labels)]
                label_prefix = f"({label_char}) "
                label_index += 1

            out.append(label_prefix + emit)

            # 뒤 공백 정리: 원문의 연속 스페이스를 소비하고, 줄바꿈/끝 아니면 1칸
            next_idx = m.end()
            while next_idx < len(t) and t[next_idx] == " ":
                next_idx += 1
            next_ch = t[next_idx] if next_idx < len(t) else ""
            if next_ch and next_ch != "\n":
                out.append(" ")

            last_index = next_idx
        else:
            out.append(m.group(0))
            last_index = m.end()

    out.append(t[last_index:])

    result = "".join(out)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result

def _normalize_plain_segment(seg: str) -> str:
    """
    괄호 밖 일반 구간:
      - 쉼표로 split -> trim -> 내부의 / 공백 정규화 -> ' / '로 join
      - 이중 공백 수축
    """
    parts = seg.split(",")
    parts = [p.strip() for p in parts]
    parts = [re.sub(r"\s*/\s*", " / ", p) for p in parts]
    joined = " / ".join([p for p in parts if p])
    joined = re.sub(r"\s{2,}", " ", joined)
    return joined


def replace_commas_with_slashes(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""

    # 0) 따옴표 정리
    t = re.sub(r"[“”]", '"', text)
    t = re.sub(r"[‘’]", "'", t)

    # 1) '[보기]' 제거
    t = re.sub(r"^\[보기\]\s*", "", t)

    # 2) [ 다음에 영어가 오는 첫 구간부터 보존 (그 전부 삭제)
    #    (^.*?(?=\[[A-Za-z]))
    t = re.sub(r"^.*?(?=\[[A-Za-z])", "", t)

    # 얕은 괄호만 (중첩 제외)
    bracket_re = re.compile(r"(\(|\[)([^()\[\]]+?)(\)|\])")

    out = []
    last_index = 0

    for m in bracket_re.finditer(t):
        open_b, inner, close_b = m.group(1), m.group(2), m.group(3)

        # (A) 괄호 이전 일반 구간 처리 (쉼표 -> /, 슬래시 공백 정규화)
        plain = t[last_index:m.start()]
        out.append(_normalize_plain_segment(plain))

        # (B) 괄호 구간 처리
        emit = m.group(0)
        is_square = (open_b == "[" and close_b == "]")
        comma_count = inner.count(",")

        if open_b == "(" and close_b == ")" and comma_count >= 2:
            # () 내부 쉼표 2개 이상 -> [] + 토큰 ' / ' 통일
            tokens = [s.strip() for s in inner.split(",") if s.strip()]
            emit = f"[ {' / '.join(tokens)} ]"
            is_square = True

        elif is_square:
            # [] 내부 정규화(, / 혼용 모두 수용)
            tokens = [s.strip() for s in re.split(r"[/,]", inner) if s.strip()]
            if len(tokens) >= 2:
                emit = f"[ {' / '.join(tokens)} ]"
            else:
                clean_inner = re.sub(r"\s+", " ", inner).strip()
                emit = f"[ {clean_inner} ]"


        if is_square:
            # 대괄호 앞/뒤 공백: 정확히 1칸(줄바꿈/시작 제외)
            current = "".join(out)

            # out 끝 연속 스페이스 제거
            while current.endswith(" "):
                current = current[:-1]
            out = [current]

            if current and not current.endswith("\n"):
                out.append(" ")

            out.append(emit)

            # 원문에서 대괄호 직후 연속 스페이스는 소비
            next_idx = m.end()
            while next_idx < len(t) and t[next_idx] == " ":
                next_idx += 1

            next_ch = t[next_idx] if next_idx < len(t) else ""
            if next_ch and next_ch != "\n":
                out.append(" ")

            last_index = next_idx
        else:
            # 대괄호 변환 대상 아니면 그대로
            out.append(emit)
            last_index = m.end()

    # (C) 남은 꼬리 일반 구간 처리
    tail = t[last_index:]
    out.append(_normalize_plain_segment(tail))

    result = "".join(out)

    # 3) 처음에 [ 또는 ( 있으면 제거
    result = re.sub(r"^[\[\(]\s*", "", result)

    # 4) 마지막에 ] 또는 ) 있으면 제거
    result = re.sub(r"\s*[\]\)]\s*$", "", result)

    # 5) 전역 이중 공백 수축 + 트림
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result

def _base_code(use_lowercase: bool) -> int:
    return ord("a") if use_lowercase else ord("A")

def _label_char(idx: int, use_lowercase: bool) -> str:
    return chr(_base_code(use_lowercase) + (idx % 26))

def label_blanks_v2(text: str, *, use_lowercase: bool = False) -> str:
    """
    Port of labelBlanksV2Upper + lowercase option.
    - Adds (A)/(a) labels before blanks, normalizes blanks to TEN underscores.
    - Supports:
      1) (A) alone -> append TEN
      2) underline bundles: _ _ _ -> label + TEN TEN...
      3) independent underline
      4) word_____ pattern
    """
    if not isinstance(text, str) or not text:
        return ""

    t = text

    # quotes / invisible spaces normalize
    t = re.sub(r"[“”]", '"', t)
    t = re.sub(r"[‘’]", "'", t)
    t = re.sub(r"[\u00A0\u2007\u202F]", " ", t)

    # helper regex depending on case
    label_re = r"\([a-z]\)" if use_lowercase else r"\([A-Z]\)"
    label_end_re = re.compile(label_re + r"\s*$")
    label_any_re = re.compile(label_re)

    # [조건 1] (라벨만) -> 뒤에 TEN 붙이기 (이미 언더라인이 뒤에 있으면 제외)
    if use_lowercase:
        t = re.sub(r"\(([a-z])\)(?!\s*_{2,})", lambda m: m.group(0) + TEN, t)
    else:
        t = re.sub(r"\(([A-Z])\)(?!\s*_{2,})", lambda m: m.group(0) + TEN, t)

    # ───────── 조건 2: 언더라인 세트 _+ (spaces) _+ ... ─────────
    src1 = t
    labels_added_in2 = 0
    bundle_anylen = re.compile(r"_+(?:\s+_+)+")  # _ _ _ 형태

    def repl_bundle(m: re.Match) -> str:
        nonlocal labels_added_in2
        run = m.group(0)
        offset = m.start()
        before = src1[:offset]
        has_prev_label = bool(label_end_re.search(before))

        groups = re.split(r"\s+", run.strip())
        normalized = " ".join([TEN for _ in groups])

        if has_prev_label:
            return MARK_L + normalized + MARK_R

        existing_before = len(label_any_re.findall(before))
        label_index = existing_before + labels_added_in2
        lc = _label_char(label_index, use_lowercase)
        labels_added_in2 += 1

        # 원본 로직: TEN으로 시작하면 (A)__________ 처럼 붙임
        label_out = f"({lc})" + normalized
        return MARK_L + label_out + MARK_R

    t = bundle_anylen.sub(repl_bundle, t)

    # ───────── 조건 3: 독립 언더라인 ─────────
    src2 = t
    labels_added_in3 = 0
    underline_re = re.compile(r"_{2,}")

    def repl_under(m: re.Match) -> str:
        nonlocal labels_added_in3
        run = m.group(0)
        offset = m.start()

        # 번들 마커 내부면 스킵
        leftL = src2.rfind(MARK_L, 0, offset + 1)
        leftR = src2.rfind(MARK_R, 0, offset + 1)
        in_marked_bundle = leftL > leftR
        if in_marked_bundle:
            return run

        before = src2[:offset]
        prev_char = before[-1] if before else ""
        if re.match(r"[A-Za-z]", prev_char):
            return run

        has_prev_label = bool(label_end_re.search(before))
        existing_before = len(label_any_re.findall(before))

        if has_prev_label:
            return TEN

        label_index = existing_before + labels_added_in2 + labels_added_in3
        lc = _label_char(label_index, use_lowercase)
        labels_added_in3 += 1
        return f"({lc})" + TEN

    t = underline_re.sub(repl_under, t)

    # ───────── 조건 4: 영문 + 언더라인 ─────────
    src3 = t
    labels_added_in4 = 0
    word_under = re.compile(r"([A-Za-z]+)_{1,}((?:\s+_{2,})*)")

    def repl_word_under(m: re.Match) -> str:
        nonlocal labels_added_in4
        run = m.group(0)
        word = m.group(1)
        tail = m.group(2) or ""
        offset = m.start()

        # 번들 마커 내부면 스킵
        leftL = src3.rfind(MARK_L, 0, offset + 1)
        leftR = src3.rfind(MARK_R, 0, offset + 1)
        in_marked_bundle = leftL > leftR
        if in_marked_bundle:
            return run

        if "_" not in run:
            return run

        before = src3[:offset]
        has_prev_label = bool(label_end_re.search(before))
        existing_before = len(label_any_re.findall(before))

        extra_groups = []
        if tail.strip():
            extra_groups = [TEN for _ in re.split(r"\s+", tail.strip()) if _]

        normalized = word + TEN + ((" " + " ".join(extra_groups)) if extra_groups else "")

        if has_prev_label:
            return normalized

        label_index = existing_before + labels_added_in2 + labels_added_in3 + labels_added_in4
        lc = _label_char(label_index, use_lowercase)
        labels_added_in4 += 1
        return f"({lc}) " + normalized

    t = word_under.sub(repl_word_under, t)

    # 마커 제거
    t = t.replace(MARK_L, "").replace(MARK_R, "")
    return t

@register_action("1. 주어진 문장 원기호 변경")
def action_wrap_circle_numbers(text: str, params: dict) -> WorkResult:
    strong_brackets = bool(params.get("strong_brackets", True))
    out = wrap_circle_numbers_clean(text, strong_brackets=strong_brackets)
    return WorkResult(
        ok=True,
        title="원기호/원문자 괄호 통일 결과",
        output_text=out,
        data={"strong_brackets": strong_brackets},
    )

@register_action("2. 정답 라벨 정렬 (A/a 선택)")
def action_format_with_labels(text: str, params: dict) -> WorkResult:
    lowercase = bool(params.get("lowercase", False))
    out = format_with_labels(text, lowercase=lowercase)
    # 라벨 사이 간격을 항상 4칸으로 강제
    out = re.sub(r"\)\s*\(", ")    (", out)

    return WorkResult(
        ok=True,
        title="정답 라벨 정렬 결과",
        output_text=out,
        data={
            "label_case": "lowercase" if lowercase else "uppercase",
            "spacing": "4 spaces",
        },
    )
    
@register_action("3. 양자택일 괄호 변경 + 라벨 부여")
def action_add_labels_to_brackets(text: str, params: dict) -> WorkResult:
    use_lowercase = bool(params.get("use_lowercase", False))
    out = add_labels_to_brackets(text, use_lowercase=use_lowercase)

    return WorkResult(
        ok=True,
        title="양자택일 괄호 변경 결과",
        output_text=out,
        data={"use_lowercase": use_lowercase},
    )

@register_action("4. 괄호 안 단어 배열 (,/→ / + ()→[] 규칙)")
def action_convert_commas_in_brackets(text: str, params: dict) -> WorkResult:
    out = convert_commas_in_brackets(text)
    return WorkResult(
        ok=True,
        title="괄호 내부 단어 배열 결과",
        output_text=out,
        data={"rule": "(),[] shallow only; (,)>=2 -> [] and join with ' / '; [] normalize separators and spacing"},
    )

@register_action("5. 보기 단어배열 (쉼표→ /, 대괄호 정규화)")
def action_replace_commas_with_slashes(text: str, params: dict) -> WorkResult:
    out = replace_commas_with_slashes(text)
    return WorkResult(
        ok=True,
        title="보기 단어배열 결과",
        output_text=out,
        data={"rule": "strip [보기], keep from first [A...], normalize commas/slashes, bracket rules applied"},
    )

@register_action("6. 밑줄 앞 기호 붙이기 (A/a 선택)")
def action_label_blanks(text: str, params: dict) -> WorkResult:
    use_lowercase = bool(params.get("use_lowercase", False))

    out = label_blanks_v2(
        text,
        use_lowercase=use_lowercase,
    )

    return WorkResult(
        ok=True,
        title="밑줄 라벨링 결과",
        output_text=out,
        data={"use_lowercase": use_lowercase},
    )

@register_action("7. 본문 단어배열 서식적용 및 밑줄")
def action_convert_commas_in_brackets_with_underline(text: str, params: dict) -> WorkResult:
    add_labels = bool(params.get("label_brackets", True))
    use_lowercase = bool(params.get("label_lowercase", False))

    out = convert_commas_in_brackets_with_underline(
        text,
        add_labels=add_labels,
        use_lowercase=use_lowercase,
    )
    return WorkResult(
        ok=True,
        title="본문 단어배열 서식 및 밑줄 적용 결과",
        output_text=out,
        data={
            "rule": "(),[] shallow only; (,)>=2 -> [] with underline; [] normalize separators and underline contents",
            "allow_underline_html": True,
            "labels_added": add_labels,
            "labels_case": "lowercase" if use_lowercase else "uppercase",
        },
    )

def render_en_work_tab(tab, st, *, review_english_text=None):
    """
    app.py에서 호출 예:
      from features.en_work import render_en_work_tab
      render_en_work_tab(tab_en_work, st, review_english_text=review_english_text)

    review_english_text는 현재는 사용하지 않지만,
    나중에 '변환 후 검수' 같은 확장용으로 인자만 유지.
    """
    with tab:
        st.subheader("🧰 영어 작업 (EN Work)")
        st.caption("변환 기능 선택 → 미리보기 확인 → 실행 → 결과 편집/저장")

        # -------------------------
        # 입력
        # -------------------------
        src_text = st.text_area(
            "입력 텍스트",
            height=220,
            key="en_work_input",
            placeholder="여기에 문제/보기/본문 텍스트를 붙여넣어 주세요.",
        )

        if not ACTIONS:
            st.error("등록된 ACTIONS가 없습니다. en_work.py에서 register_action(...)이 제대로 등록됐는지 확인해주세요.")
            return

        action_key = st.selectbox(
            "작업 선택",
            options=list(ACTIONS.keys()),
            key="en_work_action",
        )

        # -------------------------
        # 액션별 옵션
        # -------------------------
        params: Dict[str, Any] = {}

        # (2) 라벨 정렬 (A/a)
        if "라벨" in action_key and "정렬" in action_key:
            label_case = st.radio(
                "라벨 형태",
                ["대문자 (A, B, C)", "소문자 (a, b, c)"],
                horizontal=True,
                key="en_work_label_case",
            )
            params["lowercase"] = label_case.startswith("소문자")

        # (3) 양자택일 괄호 + 라벨
        if "양자택일" in action_key or ("괄호" in action_key and "라벨" in action_key):
            label_case2 = st.radio(
                "라벨 형태",
                ["대문자 (A, B, C)", "소문자 (a, b, c)"],
                horizontal=True,
                key="en_work_bracket_label_case",
            )
            params["use_lowercase"] = label_case2.startswith("소문자")

        # (6) 밑줄 라벨
        if ("밑줄" in action_key or "blank" in action_key.lower()) and action_key != "7. 본문 단어배열 서식적용 및 밑줄":
            label_case3 = st.radio(
                "라벨 형태",
                ["대문자 (A, B, C)", "소문자 (a, b, c)"],
                horizontal=True,
                key="en_work_blank_label_case",
            )
            params["use_lowercase"] = label_case3.startswith("소문자")

        params["strong_brackets"] = st.checkbox(
            "[...]를 <strong>로 감싸기 (모든 기능에 적용)",
            value=True,
            key="en_work_strong_brackets",
        )

        # (7) 본문 단어배열 서식+밑줄: 라벨 옵션 (3번과 동일 UX)
        if action_key == "7. 본문 단어배열 서식적용 및 밑줄":
            label_case_7 = st.radio(
                "[] 라벨 형태",
                ["대문자 (A, B, C)", "소문자 (a, b, c)"],
                horizontal=True,
                key="en_work_7_label_case",
            )
            params["label_brackets"] = True
            params["label_lowercase"] = label_case_7.startswith("소문자")

        # -------------------------
        # 미리보기
        # -------------------------
        with st.expander("🔎 미리보기", expanded=False):
            auto_preview = st.checkbox("입력할 때마다 자동 미리보기", value=True, key="en_work_auto_preview")

            preview_result: Optional[WorkResult] = None
            if auto_preview and src_text.strip():
                preview_result = run_action(action_key, src_text, params)

                if preview_result.ok:
                    html_in, html_out = highlight_diff_html(src_text, preview_result.output_text)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**입력(변경점 표시)**")
                        st.markdown(wrap_pre_block(html_in), unsafe_allow_html=True)
                    with c2:
                        st.markdown("**출력 미리보기(변경점 표시)**")
                        st.markdown(wrap_pre_block(html_out), unsafe_allow_html=True)
                else:
                    st.warning(f"미리보기 실패: {preview_result.error}")

        # -------------------------
        # 실행/초기화
        # -------------------------
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            run = st.button("실행", type="primary", key="en_work_run")
        with c2:
            reset = st.button("결과 초기화", key="en_work_reset")
        with c3:
            st.caption("※ 실행하면 아래에 ‘편집 가능한 결과’가 생성됩니다.")

        if reset:
            st.session_state.pop("en_work_result", None)
            st.session_state.pop("en_work_edit", None)
            st.session_state.pop("en_work_error", None)
            st.session_state.pop("en_work_edit_area", None)
            st.session_state.pop("en_work_result_text", None)
            st.rerun()

        if run:
            st.session_state.pop("en_work_error", None)

            if not src_text.strip():
                st.warning("먼저 입력 텍스트를 넣어주세요.")
            else:
                with st.spinner("처리 중..."):
                    result = run_action(action_key, src_text, params)
                st.session_state["en_work_result"] = result

                if result.ok:
                    # 편집 가능한 버퍼 생성
                    st.session_state["en_work_edit"] = result.output_text
                    # 기존 위젯 상태 초기화 후 새 값으로 시작하도록 키 제거
                    st.session_state.pop("en_work_edit_area", None)
                    st.session_state["en_work_result_text"] = result.output_text
                else:
                    st.session_state["en_work_error"] = result.error

        # -------------------------
        # 결과 표시 + 편집
        # -------------------------
        
        result: Optional[WorkResult] = st.session_state.get("en_work_result")
        if not result:
            st.info("위에서 기능을 선택하고 ‘실행’을 누르면 결과가 나옵니다.")
            return

        if not result.ok:
            st.error(result.error)
            return

        # 최신 편집 버퍼를 우선 사용 (text_area가 리렌더되며 en_work_edit_area에 저장된 값이 있으면 반영)
        current_edit = st.session_state.get(
            "en_work_edit_area",
            st.session_state.get("en_work_edit", result.output_text),
        )
        st.session_state["en_work_edit"] = current_edit

        # 새 실행 결과가 이전과 다르면 편집 버퍼를 최신 실행 결과로 동기화
        last_result_text = st.session_state.get("en_work_result_text")
        if result.output_text != last_result_text:
            st.session_state["en_work_edit"] = result.output_text
            st.session_state["en_work_result_text"] = result.output_text
            st.session_state.pop("en_work_edit_area", None)

        action_current = st.session_state.get("en_work_action", action_key)

        st.markdown("### 📌 최종본 (강조 렌더링)")
        st.info("**태그(굵게 등)가 포함된 부분은 드래그해서 직접 복사·붙여넣기 해주세요.**")
        final_text = st.session_state.get("en_work_edit", result.output_text) or ""

        copy_payload = json.dumps(final_text)

        # 최종본: 강조(strong)만 렌더, 밑줄 태그는 그대로 표시
        render_fn = render_strong_html
        st.markdown(
            "<div id='en_final_render_box' style='background:#f7f7f7; border:1px solid #e5e5e5; "
            "border-radius:8px; padding:12px; line-height:1.8; "
            "font-weight:400; white-space: pre-wrap;'>"
            "<style> strong{font-weight:800;} u{text-decoration-thickness:2px;} </style>"
            f"{render_fn(final_text)}"
            "</div>",
            unsafe_allow_html=True,
        )
        # 최종본 텍스트 코드 블록 + 복사 버튼 (components: 안정적 실행)
        st.components.v1.html(
            f"""
            <div style="display:flex; align-items:center; gap:8px; margin: 8px 0;">
              <div style="font-weight:600;">📄 최종본 텍스트</div>
              <button id="copy_btn" type="button"
                style="padding:4px 8px; border-radius:6px; border:1px solid #ddd; background:#f5f5f5; cursor:pointer;">
                복사
              </button>
              <span id="copy_msg" style="font-size:12px; color:#666;"></span>
            </div>
            <pre style="white-space:pre-wrap; background:#f7f7f7; border:1px solid #e5e5e5; border-radius:8px; padding:12px; max-height:240px; overflow:auto;">{html.escape(final_text)}</pre>
            <script>
              const text = {copy_payload};
              const btn = document.getElementById("copy_btn");
              const msg = document.getElementById("copy_msg");

              async function doCopy() {{
                try {{
                  await navigator.clipboard.writeText(text);
                  msg.textContent = "복사 완료!";
                  setTimeout(()=>msg.textContent="", 1200);
                }} catch (e) {{
                  msg.textContent = "복사 실패";
                  console.error(e);
                }}
              }}

              if (btn) {{
                btn.addEventListener("click", doCopy);
              }}
            </script>
            """,
            height=260,
            scrolling=True,
        )

        # 위젯 초기값 동기화: 새로운 실행 결과가 있으면 widget state를 초기화
        if "en_work_edit_area" not in st.session_state:
            st.session_state["en_work_edit_area"] = st.session_state.get("en_work_edit", result.output_text)

        edited = st.text_area(
            "아래 텍스트를 직접 수정할 수 있어요 (이 값이 최종본이 됩니다).",
            height=220,
            value=st.session_state.get("en_work_edit", result.output_text),
            key="en_work_edit_area",
        )
        # text_area 값으로 편집 버퍼 업데이트 (복사/다운로드와 동기화)
        st.session_state["en_work_edit"] = edited
        edit_copy_payload = json.dumps(edited)

        st.markdown(
            """
            <script>
            // 결과 편집 헤더 복사 버튼 제거 -> 자동 textarea 복사 버튼만 사용
            </script>
            """,
            unsafe_allow_html=True,
        )

        # 모든 textarea에 복사 버튼 자동 부착(JS) - ko_work와 동일 UX
        st.markdown(
            """
            <script>
            const attachEnCopyButtons = () => {
              const areas = document.querySelectorAll('textarea[data-testid="stTextArea"]');
              areas.forEach((ta) => {
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
            window.addEventListener('load', attachEnCopyButtons);
            setTimeout(attachEnCopyButtons, 500);
            </script>
            """,
            unsafe_allow_html=True,
        )

        csave, cdl, ccopy = st.columns([1, 1, 2])
        with csave:
            if st.button("수정본 저장", key="en_work_save_edit"):
                st.session_state["en_work_edit"] = edited
                st.success("수정본을 저장했습니다. (아래 ‘최종본’이 업데이트됩니다.)")

        with cdl:
            st.download_button(
                "최종본 다운로드(.txt)",
                data=(st.session_state.get("en_work_edit", edited) or ""),
                file_name="en_work_result.txt",
                mime="text/plain",
                key="en_work_download",
            )

        with ccopy:
            st.caption("※ Streamlit은 ‘클립보드 복사’ 버튼이 기본 제공되지 않아, 텍스트를 드래그해서 복사하면 됩니다.")

        st.markdown(
            f"""
            <script>
            const btnEditEn = document.getElementById("en_edit_copy_btn");
            if (btnEditEn) {{
              btnEditEn.onclick = async () => {{
                try {{
                  await navigator.clipboard.writeText({edit_copy_payload});
                  const old = btnEditEn.innerText;
                  btnEditEn.innerText = "복사 완료!";
                  setTimeout(()=>{{btnEditEn.innerText = old;}}, 1200);
                }} catch(e) {{
                  btnEditEn.innerText = "복사 실패";
                }}
              }};
            }}
            </script>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### ✅ 실행 결과")
        st.caption(result.title)

        # diff 하이라이트 (실행 결과 기준)
        html_in2, html_out2 = highlight_diff_html(src_text, result.output_text)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**입력(변경점 표시)**")
            st.markdown(wrap_pre_block(html_in2), unsafe_allow_html=True)
        with cc2:
            st.markdown("**실행 출력(변경점 표시)**")
            st.markdown(wrap_pre_block(html_out2), unsafe_allow_html=True)



        if result.data:
            with st.expander("디버그 데이터", expanded=False):
                st.json(result.data, expanded=False)
