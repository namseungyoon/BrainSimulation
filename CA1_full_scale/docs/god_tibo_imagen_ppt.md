---
marp: true
title: god-tibo-imagen Technical Brief
description: Codex private image-generation backend wrapper rationale, installation, architecture, and operating guidance
paginate: true
---

# god-tibo-imagen 도입 검토 자료

Codex ChatGPT 인증 기반 이미지 생성 래퍼의 목적, 동작 원리, 설치 절차, 구현 방식, 운영 리스크

Prepared for HITL decision

---

# 1. 한 장 요약

`god-tibo-imagen`은 로컬 Codex ChatGPT 로그인 상태를 재사용해 이미지 생성을 호출하는 Node.js CLI/라이브러리 + Python SDK + Agent Skill 패키지다.

핵심 가치는 “에이전트가 텍스트/참조 이미지로 PNG 산출물을 자동 생성하고 저장하는 표준 실행면”을 제공하는 것이다.

단, 이 패키지는 공식 공개 API 클라이언트가 아니다. 저장소 README와 CLI가 모두 경고하듯 `chatgpt.com/backend-api/codex/responses`라는 private Codex backend path에 의존한다. 따라서 연구/내부 자동화/프로토타입에는 유용하지만, 장기 운영·제품·컴플라이언스 환경에는 강한 리스크가 있다.

---

# 2. 왜 이 기법을 고려하나

에이전트 작업에서 이미지 생성이 필요한 경우가 반복된다.

- 프론트엔드 mockup, 아이콘, 설명 이미지, 보고서 삽화 생성
- 기존 이미지를 입력으로 주고 변형/합성/스타일 적용
- CLI, Node, Python, Agent Skill에서 같은 이미지 생성 경로를 재사용
- 산출물을 파일로 남겨 downstream 문서/앱/QA에 연결

일회성 수동 이미지 생성 대신, prompt -> request -> streamed response -> PNG 저장 -> JSON summary라는 자동화 가능한 파이프라인을 만든다는 점이 도입 이유다.

---

# 3. 무엇을 “기법”이라고 부르는가

이 기법의 본질은 이미지 모델 자체가 아니라 “Codex/ChatGPT 인증 상태를 가진 로컬 에이전트가 이미지 생성 backend를 호출하고 결과 PNG를 artifact로 저장하는 adapter pattern”이다.

구성 요소:

- 인증 재사용: `~/.codex/auth.json`, `~/.codex/installation_id`
- 요청 생성: Codex `/responses` body + `image_generation` tool
- 스트리밍 처리: SSE event stream parsing
- 결과 추출: `image_generation_call.result` base64 PNG
- 저장: base64 decode 후 PNG 파일 저장
- 대체 경로: `codex exec`가 생성한 `~/.codex/generated_images/` PNG 탐색
- Agent Skill: 다양한 coding agent가 동일 방식으로 호출

---

# 4. 설치/검증 결과

검토 시점 설치 검증:

```bash
git clone --depth 1 https://github.com/NomaDamas/god-tibo-imagen.git /tmp/god-tibo-imagen
cd /tmp/god-tibo-imagen
npm ci
npm test
npm run check
node src/cli/generate.js --version
node src/cli/generate.js --help
```

확인 결과:

- `npm ci`: 의존성 설치 성공, vulnerability 0
- `npm test`: 41 tests pass
- `npm run check`: syntax check passed for 28 files
- CLI version: `0.3.1`
- live image generation은 의도적으로 실행하지 않음

---

# 5. 설치 옵션

Node CLI 전역 설치:

```bash
npm install -g god-tibo-imagen
gti --version
gti --help
```

Node library:

```bash
npm install god-tibo-imagen
```

Python SDK:

```bash
pip install god-tibo-imagen
```

Agent Skill:

```bash
npx skills add NomaDamas/god-tibo-imagen --skill god-tibo-imagen
```

---

# 6. 필수 전제

이 패키지는 자체 인증 시스템을 제공하지 않는다.

필수:

- Node.js 20+
- Python SDK 사용 시 Python 3.10+
- 로컬 Codex ChatGPT 로그인 상태
- `~/.codex/auth.json` 존재
- 가능하면 `~/.codex/installation_id` 존재
- 계정이 private backend image generation을 사용할 권한 보유

인증 파일이 없거나 access token/account id가 없으면 private provider는 실패한다. 검토 중 의도적으로 `/tmp/no-auth.json`을 넣었을 때 `ENOENT`로 실패해 인증 파일 의존성이 확인됐다.

---

# 7. 기본 사용 흐름

CLI:

```bash
gti \
  --provider private-codex \
  --prompt "flat blue square icon" \
  --output ./out.png
```

참조 이미지:

```bash
gti \
  --prompt "Make this cat wear a hat" \
  --image ./cat.png \
  --output ./cat-hat.png
```

dry run:

```bash
gti --prompt "flat blue square icon" --dry-run
```

---

# 8. 전체 아키텍처

```text
User / Agent
  |
  v
gti CLI / Node API / Python SDK / Agent Skill
  |
  v
resolveConfig()
  |
  v
Provider selection: private-codex | codex-cli | auto
  |
  +--> private-codex:
  |      load auth -> build /responses request -> fetch SSE
  |      -> parse SSE -> extract image_generation_call -> save PNG
  |
  +--> codex-cli:
         codex login status -> codex exec prompt
         -> scan ~/.codex/generated_images -> copy newest PNG
```

---

# 9. 모듈 책임 분리

주요 파일 기준:

- `src/config.js`: 기본 경로, provider, model, originator, output path 결정
- `src/auth/loadCodexSession.js`: Codex auth/installation id 로드
- `src/auth/validateSession.js`: access token/account id/auth mode 검증
- `src/codex/buildResponsesRequest.js`: private `/responses` 요청 구성
- `src/codex/streamResponsesSse.js`: SSE stream parsing
- `src/codex/extractImageGeneration.js`: 이미지 생성 결과 추출
- `src/fs/saveImage.js`: base64 PNG 저장
- `src/providers/privateCodexProvider.js`: private HTTP provider
- `src/providers/codexCliProvider.js`: `codex exec` fallback provider
- `src/cli/generate.js`: CLI argument parsing and execution

---

# 10. private-codex provider 내부 절차

1. 설정 해석
   - `CODEX_HOME`, auth file, installation id file, base URL, model, provider

2. 세션 로드
   - `auth.json`에서 `tokens.access_token`, `tokens.account_id` 추출
   - `installation_id`가 있으면 client metadata에 사용

3. 세션 검증
   - access token 누락 시 fail
   - account id 누락 시 fail
   - `auth_mode != chatgpt`면 warning
   - JWT expiration은 warning

4. 요청 body 생성
   - `tools: [{ type: "image_generation", output_format: "png" }]`
   - `stream: true`
   - prompt와 optional image input 포함

---

# 11. private request shape

요청 endpoint:

```text
POST https://chatgpt.com/backend-api/codex/responses
Accept: text/event-stream
Authorization: Bearer <access_token>
ChatGPT-Account-ID: <account_id>
originator: codex_cli_rs
session_id: <uuid>
```

핵심 body:

```json
{
  "model": "gpt-5.4",
  "input": [{ "type": "message", "role": "user", "content": [...] }],
  "tools": [{ "type": "image_generation", "output_format": "png" }],
  "tool_choice": "auto",
  "parallel_tool_calls": false,
  "stream": true,
  "store": false
}
```

---

# 12. 이미지 입력 처리

CLI의 `--image`는 파일을 읽어서 data URL로 변환한다.

지원 확장자:

- `png`
- `jpg`
- `jpeg`
- `gif`
- `webp`

요청 content에는 다음 형태로 들어간다.

```json
{ "type": "input_image", "image_url": "data:image/png;base64,..." }
```

debug dump에서는 이미지 data URL이 `[REDACTED_IMAGE_DATA]`로 치환된다.

---

# 13. output size 처리

지원 size:

- `auto`
- `1024x1024`, `2048x2048`
- `1536x1024`, `2048x1152`, `3840x2160`
- `1024x1536`, `2160x3840`

private provider는 size를 `image_generation` tool config에 전달한다.

중요: `codex-cli` provider는 size를 보장하지 못한다. 따라서 `auto` provider에서 private path가 실패했더라도 `--size`가 있으면 codex-cli fallback을 거부한다. 치수 보장을 조용히 깨지 않기 위한 fail-fast 설계다.

---

# 14. SSE 처리 방식

private backend는 streamed SSE를 반환할 수 있다.

처리 단계:

1. stream text를 event block으로 분리
2. 각 block의 `event:`와 `data:` line 파싱
3. `data:` JSON parse
4. `response.created`, `response.output_item.done`, `response.completed` 추적
5. 완료된 output item 중 `image_generation_call`을 찾음

fixture 기반 테스트는 정상 SSE, malformed SSE, partial image event, no-image case를 검증한다.

---

# 15. 이미지 추출 규칙

우선순위:

1. `response.output_item.done` item 중 마지막 `image_generation_call` + `result`
2. 없으면 `response.image_generation_call.partial_image`의 `partial_image_b64`
3. 둘 다 없으면 `MISSING_IMAGE_GENERATION_OUTPUT`

저장:

```text
base64 PNG string
  -> validate standard base64
  -> Buffer.from(..., "base64")
  -> mkdir parent
  -> writeFile(outputPath)
```

data URL이 결과로 오면 저장하지 않고 거부한다. 결과는 raw base64 PNG bytes여야 한다.

---

# 16. codex-cli fallback provider

fallback은 private HTTP path가 불안정할 때 `codex exec`를 우회 경로로 사용한다.

절차:

1. `codex --version`
2. `codex login status`
3. ChatGPT 로그인 확인
4. 임시 디렉터리 생성
5. `codex exec --ephemeral --sandbox workspace-write ...`
6. session id 추출
7. `~/.codex/generated_images/<session>/` 또는 최근 PNG 탐색
8. 찾은 PNG를 사용자가 지정한 output path로 복사

한계:

- input image 미지원
- output size 선택 미지원
- 생성 결과 탐색이 파일시스템 관찰 기반

---

# 17. provider 선택 전략

`private-codex`

- 가장 직접적인 경로
- size, image input 지원
- private backend 변경에 취약

`codex-cli`

- 공식 Codex CLI 실행면을 빌려 쓰는 우회 경로
- image input/size 제약
- 생성 파일 탐색 필요

`auto`

- private-codex 먼저 시도
- 실패 시 codex-cli fallback
- 단, size가 있으면 fallback 거부

권장: 자동화/실험은 `auto`, deterministic artifact/치수 요구는 `private-codex` with fail-fast.

---

# 18. 왜 이 설계가 유용한가

1. 에이전트 친화성
   - CLI, Node API, Python SDK, Agent Skill을 모두 제공

2. 산출물 중심
   - 최종 결과가 PNG 파일로 저장되고 JSON summary로 경로를 반환

3. dry-run 가능
   - live network call 없이 request shape 확인 가능

4. debug 가능
   - request/response metadata dump를 남기되 secret/image payload는 redaction

5. fallback 가능
   - private HTTP path와 codex-cli path를 분리

6. 테스트 가능
   - SSE parsing, auth validation, image extraction, save path를 fixture로 검증

---

# 19. 왜 조심해야 하나

가장 큰 리스크는 지원 계약 부재다.

README와 CLI warning:

```text
This project calls an unsupported private Codex backend path.
The contract may break without notice.
```

위험:

- private endpoint path 변경
- request schema 변경
- SSE event shape 변경
- auth file schema 변경
- 계정 entitlement 변경
- ChatGPT/Codex 제품 정책 변경
- 조직 보안 정책과 충돌 가능

따라서 이 도구는 “내부 자동화/프로토타입/개인 생산성”에는 적합하지만, 고객-facing 제품의 안정 API로 간주하면 안 된다.

---

# 20. 보안 관점

민감 자산:

- `~/.codex/auth.json`
- access token
- refresh token
- account id
- installation id
- generated image payload

현재 방어:

- debug header에서 `Authorization` redaction
- account id/session id/installation id redaction
- input image data URL redaction
- response image base64 redaction
- missing/invalid auth fail-fast

운영 규칙:

- auth 파일을 로그/이슈/PR에 첨부 금지
- debug dump도 외부 공유 전 재검토
- CI에서 real auth 사용 금지
- live smoke는 개인/격리 환경에서만 수행

---

# 21. 테스트 설계

검토 시점 테스트 범위:

- Agent Skill layout/frontmatter
- auth load/validation
- CLI version/help behavior
- codex-cli fallback image discovery/copy
- fallback 제한: image input, size
- image generation extraction
- base64 PNG save validation
- public library exports
- auto provider fallback
- private provider SSE success path
- request builder shape
- unsupported size rejection
- debug redaction
- malformed SSE handling

결과: 41 tests pass.

---

# 22. 권장 운영 Runbook

초기 확인:

```bash
gti --version
gti --help
gti --prompt "flat blue square icon" --dry-run
```

실제 생성:

```bash
gti \
  --provider private-codex \
  --prompt "..." \
  --output ./artifact.png \
  --debug \
  --debug-dir ./.debug-gti
```

실패 시:

1. `auth.json` 존재 확인
2. `codex login status` 확인
3. `--dry-run`으로 request shape 확인
4. 401이면 Codex ChatGPT 재로그인
5. no-image면 SSE/debug artifact 확인
6. private path 불안정하면 `--provider auto` 또는 `codex-cli`

---

# 23. 의사결정 매트릭스

| 상황 | 권장 |
|---|---|
| 개인/내부 agent workflow에서 이미지 파일을 빠르게 만들기 | 사용 적합 |
| prompt/image input/size를 코드에서 제어해야 함 | private-codex 적합 |
| private path 실패 시 우회가 필요함 | auto 적합 |
| 결과 치수가 반드시 보장돼야 함 | private-codex only, fallback 금지 |
| 고객-facing production 기능 | 부적합 |
| 보안/감사/계약 안정성이 중요 | 공식 지원 API 사용 권장 |
| CI에서 deterministic 테스트 | fixture/dry-run만 사용 |
| 조직 auth token 관리가 엄격함 | 도입 전 보안 검토 필요 |

---

# 24. 결론

`god-tibo-imagen`은 “에이전트가 이미지 생성 산출물을 파일로 남기는 자동화 adapter”로는 잘 설계되어 있다.

기술적으로는:

- auth reuse
- private request builder
- streamed SSE parser
- image extraction
- PNG save
- codex-cli fallback
- Agent Skill packaging
- redacted debug
- fixture-based tests

가 갖춰져 있다.

하지만 전략적으로는 unsupported private backend wrapper다. 따라서 이 자료의 결론은 “공식 API 대체재”가 아니라, “HITL이 리스크를 이해한 상태에서 내부 agent image-generation workflow를 빠르게 열기 위한 도구”로 채택하는 것이다.

---

# Appendix A. 확인한 소스

로컬 클론: `/tmp/god-tibo-imagen`

주요 파일:

- `README.md`
- `package.json`
- `src/config.js`
- `src/cli/generate.js`
- `src/auth/loadCodexSession.js`
- `src/auth/validateSession.js`
- `src/codex/buildResponsesRequest.js`
- `src/codex/streamResponsesSse.js`
- `src/codex/extractImageGeneration.js`
- `src/providers/privateCodexProvider.js`
- `src/providers/codexCliProvider.js`
- `src/fs/saveImage.js`
- `skills/god-tibo-imagen/SKILL.md`
- `python/README.md`

검증:

- `npm ci`
- `npm test`: 41 pass
- `npm run check`: syntax check passed
- CLI `--version`, `--help`

