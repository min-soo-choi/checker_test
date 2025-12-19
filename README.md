# 텍스트 자동 검수기 + 작업 도구

한국어/영어 검수, 국어/영어 작업 유틸, PDF 해설 텍스트 정리를 하나로 묶은 Streamlit 앱입니다.  
기능 중심으로 정리한 설계 문서 형태의 README입니다.

---

## 기능 요약

- 한국어 검수: 객관적 오류만 탐지 (오탈자/띄어쓰기/문장부호/조사·어미)
- 영어 검수: 객관적 오류만 탐지 (스펠링/대문자/중복/종결부호)
- 국어 작업: 작품 들여쓰기/줄바꿈 정리 (시트 검색/ PDF OCR)
- 영어 작업: 시험 지문/보기/문항 전처리 도구 모음
- PDF 해설 정리: `[정답 해설] / [오답 해설]` 표준화

---

## 기능 상세

### 1) 한국어 검수

**목표**
- 주관적 수정 없이 객관적 오류만 탐지
- 오류 검출 후 후처리로 오탐/환각 제거

**탐지 범위**
- 오탈자, 조사/어미 오류
- 명백한 띄어쓰기 오류
- 문장부호 오류(종결부호 누락, 따옴표 짝 불일치 등)
- 옵션: 단어 내부 분리 오류(`된 다` → `된다`)

**동작 흐름**
1. 한국어 전용 프롬프트 생성
2. Gemini(JSON mode) 호출
3. Detector → Judge 구조로 필터링
4. 최종 리포트/하이라이트 출력

**출력**
- suspicion_score (1~5)
- translated_typo_report
- raw/final JSON 및 diff

---

### 2) 영어 검수

**목표**
- 의미나 문체를 바꾸지 않고 오류만 탐지

**탐지 범위**
- 스펠링 오류
- split-word 오류 (`wi th`, `o f`)
- AI/Al 혼동
- 대문자 규칙 위반
- 중복 단어
- 종결부호 누락

**출력**
- suspicion_score
- content_typo_report
- raw/final JSON 및 diff

---

### 3) 국어 작업 (KO Work)

**기능 1: 1. 시트 검색 작품 들여쓰기**
- 시: 줄바꿈만 정규화 (들여쓰기 없음)
- 시 이외: 줄바꿈 유지 + 각 줄 시작 1칸 들여쓰기

**기능 2: 2. PDF 작품 들여쓰기**
- 문학-운문: anchors 기준 줄바꿈
- 문학-산문: anchors 기준 문단 구분 + 들여쓰기
- 문학 이외: 입력 줄바꿈 유지 + 들여쓰기

**입력/출력**
- 입력: OCR 텍스트
- 출력: 줄바꿈/들여쓰기 정리된 텍스트

---

### 4) 영어 작업 (EN Work)

**전처리 기능**
- 원기호/원문자 통일 및 괄호 정리
- 정답 라벨 정렬 (A/a, 4칸 간격 보정)
- 양자택일 괄호 변경 + 라벨 부여
- 괄호 안 단어 배열 정규화
- 보기 단어배열 정리 (쉼표 → `/`)
- 밑줄 라벨링 (A/a 자동 부여)

**공통 옵션**
- `[...]`를 `<strong>`로 감싸기 (모든 기능 적용)

---

### 5) PDF 해설 텍스트 정리

**목표**
- `[정답 해설] / [오답 해설]` 형식으로 통일
- 헤더 순서 및 블록 간 줄바꿈 규칙 준수

**규칙**
- `[정답 해설]` → `[오답 해설]` 순서 보장
- 블록 사이 빈 줄 1줄 유지
- 원기호(①, ②, ㉠…)는 줄바꿈 분리
- `[오답 해설]` 내부 원기호 붙음 케이스 보정

**출력**
- code block 형태로 반환 (복사 용이)

---

## 설치 및 실행

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 환경 설정 (Streamlit Secrets)

`.streamlit/secrets.toml`에 아래 값을 설정합니다.

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

# (선택) 정산/로깅용 Google Sheets
LOG_SHEET_ID = "YOUR_SHEET_ID"
LOG_WORKSHEET = "usage_log_worker"
GCP_SERVICE_ACCOUNT_JSON = """{
  "type": "service_account",
  "project_id": "...",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "...@....iam.gserviceaccount.com",
  "client_id": "...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "..."
}"""
```

- `LOG_SHEET_ID`가 없으면 로깅은 자동 비활성화됩니다.
- 서비스 계정에 해당 시트 편집 권한이 있어야 합니다.

---

## 정산/로깅 (Google Sheets)

- 로그는 헤더 아래의 **첫 빈 행(A열 기준)**에 기록됩니다.
- 기본 컬럼: timestamp, session_id, feature, model, status, latency, token/cost 등

---

## 폴더 구조

```
app.py                  # 메인 Streamlit 앱
features/ko_work.py     # 국어 작업 기능
features/en_work.py     # 영어 작업 기능
requirements.txt
```

---

## 배포

1) GitHub에 push  
2) Streamlit Cloud에서 레포 연결 후 배포  
3) Secrets는 Streamlit Cloud 설정에서 동일하게 입력

---

## 문제 해결

- `GEMINI_API_KEY` 미설정: 앱 시작 시 에러 발생
- 시트 로그 미기록: `LOG_SHEET_ID`/서비스 계정 권한 확인
- 결과 줄바꿈 문제: PDF 원문 내 원기호/헤더 패턴을 점검

---

## 라이선스

내부 프로젝트용. 필요 시 별도 명시.
