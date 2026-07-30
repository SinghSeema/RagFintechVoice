# RagFintechVoice — Frontend Design Options

**Status:** Option A selected for immediate build (Phase 3 start).
**Last updated:** 2026-04-07

---

## Context

Two backend APIs are live:

| API | Port | Protocol | Purpose |
|-----|------|----------|---------|
| Voice agent server | 8001 | WebRTC (LiveKit) + REST | Real-time voice conversation with RAG bot |
| Text gateway | 8000 | HTTP REST | Single-shot and multi-hop text queries |

The frontend must expose both to a user in a coherent interface.

---

## Option A — Single-Page HTML (selected)

**File:** `index.html` (served by `python -m http.server 8080`)

### Layout

```
┌──────────────────────────────────────────────────────────┐
│  🏦 Finova  │  [Voice]  [Chat]                           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─── VOICE TAB ──────────────────────────────────────┐  │
│  │                                                    │  │
│  │   [ Connect & Start Agent ]  [ Disconnect ]        │  │
│  │                                                    │  │
│  │   Status pill: ● Disconnected / ● Connected /      │  │
│  │                ● Bot speaking / 🎤 You speaking     │  │
│  │                                                    │  │
│  │   Transcript feed (scrollable):                    │  │
│  │     You: "What documents does an NRI need?"        │  │
│  │     Finova: "For NRI KYC, the following …"         │  │
│  │                                                    │  │
│  │   [Unmute audio] button (shown if autoplay blocked)│  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌─── CHAT TAB ───────────────────────────────────────┐  │
│  │                                                    │  │
│  │   Mode: [Single-shot ▼]  [Multi-hop decompose]     │  │
│  │                                                    │  │
│  │   Chat bubble feed:                                │  │
│  │     ┌──────────────────────────────────────────┐   │  │
│  │     │ You: Compare simplified KYC with CDD     │   │  │
│  │     └──────────────────────────────────────────┘   │  │
│  │     ┌──────────────────────────────────────────┐   │  │
│  │     │ Finova: …answer…                         │   │  │
│  │     │ Sources: [1] Section 4.2 · Chapter 3     │   │  │
│  │     │          [2] Section 6.1 · Chapter 5     │   │  │
│  │     └──────────────────────────────────────────┘   │  │
│  │                                                    │  │
│  │   [ Type your question …          ] [Send]         │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Technical choices

| Concern | Decision | Why |
|---------|----------|-----|
| Build system | None — single `.html` file | Zero setup; served by `python -m http.server` |
| LiveKit SDK | `livekit-client.umd.js` (already vendored at repo root) | Avoids CDN; already tested in `test_client.html` |
| Styling | Inline CSS (CSS variables for theme) | No framework dependency |
| JS | Vanilla ES2022 (`async/await`, `fetch`, modules via `<script type="module">`) | No bundler needed |
| Text API calls | `fetch` to `http://localhost:8000` | Direct CORS (server allows `*`) |
| Voice transport | LiveKit SDK → `wss://` room | Token fetched from `http://localhost:8001/token` |
| Transcript | Built from LiveKit `TrackSubscribed` + STT callbacks | Displayed in real time |

### API flow — Voice tab

```
Click "Connect"
  → GET /token?room=rag-voice&identity=web-<uuid>    (port 8001)
  → POST /agent/start?room=rag-voice                 (port 8001)
  → room.connect(livekit_url, token)                 (LiveKit SDK)
  → room.startAudio()                                (user-gesture context)
  → on TrackSubscribed: attach audio track           (bot speaks)
  → on TranscriptionReceived / DataReceived: update transcript
Click "Disconnect"
  → room.disconnect()
  → POST /agent/stop?room=rag-voice                  (port 8001)
```

### API flow — Chat tab

```
User submits question
  → Single-shot: POST http://localhost:8000/query
  → Multi-hop:   POST http://localhost:8000/query/decompose
Response:
  { answer: "…", sources: [{rank, section_title, chapter, snippet}] }
Render:
  - Answer as chat bubble
  - Collapsible sources list beneath the answer
```

### Files produced

| File | Purpose |
|------|---------|
| `index.html` | Single self-contained frontend page |

### Limitations

- No auth — dev/demo only
- CORS open (`*`) — must tighten before any external deployment
- No mobile layout (portrait viewport) — Phase 3 React Native handles mobile

---

## Option B — React / Next.js SPA (deferred to Phase 3)

**Rationale for deferral:** Requires Node, build pipeline, and LiveKit React SDK setup. Option A unblocks demo and end-to-end testing immediately.

### When to choose

- React Native mobile client needs shared component logic
- Multiple engineers working on UI simultaneously
- Production deployment with auth, routing, and SSR

### Stack

| Layer | Tool |
|-------|------|
| Framework | Next.js 14 (App Router) |
| Voice UI | `@livekit/components-react` — pre-built `VoiceAssistant`, `BarVisualizer`, `TranscriptionTile` |
| HTTP client | Axios or native `fetch` |
| Styling | Tailwind CSS |
| State | Zustand (lightweight) |
| Mobile reuse | React Native + Expo (Phase 3) — shared hooks, not components |

### Key LiveKit React components

```tsx
// Voice tab
<LiveKitRoom serverUrl={livekitUrl} token={token} connect>
  <VoiceAssistant />          // waveform + bot audio
  <BarVisualizer />           // user mic level
  <TranscriptionTile />       // live transcript
</LiveKitRoom>

// Chat tab — custom component, calls text API
<ChatPanel endpoint="http://localhost:8000/query" />
```

### Migration path from Option A

1. Extract `_connectVoice()` and `_sendChat()` logic into React hooks
2. Replace transcript rendering with `<TranscriptionTile>`
3. Add Next.js routing: `/voice`, `/chat`, `/history`
4. Wire auth (Clerk or NextAuth) to token endpoint

---

## Decision log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-04-07 | Build Option A first | Unblocks demo immediately; no build toolchain needed |
| 2026-04-07 | Defer Option B | Required for Phase 3 mobile, but premature now |
