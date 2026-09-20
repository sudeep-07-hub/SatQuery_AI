# FIX_LOG.md — SatQuery AI Fix Log

*Append-only. Each fix is logged as it happens, never batched.*

Entries before 2026-09-20 were removed from the working tree by commit `9430976`
("clean the repository down to what runs the system"). They are not lost — recover them with
`git show 9430976^:FIX_LOG.md`. This file resumes the same format from that point.

Evidence for the entries below: `/Users/sukesh/Desktop/satquery-phase1-evidence/`
(`before/`, `after/`, `focus/`, each with a `measurements.json`). Kept outside the repo so it is
not committed; regenerate with the scripts noted per entry.

---

[UI] [FRONTEND] Composer row: the textarea did not match the buttons it sits beside (2026-09-20)
Scope: `frontend/src/components/assistant/ChatComposer.tsx` and the composer block of
`frontend/src/index.css`. No MC1–MC8 logic, no backend, no other component.
Symptom: the `+` button and Send were 40 px tall; the textarea was 64 px. The row is
`align-items: flex-end`, so the three boxes did share a bottom edge — but the *text* did not sit
with the buttons. Measured at 375 px and 1280 px, the centre of the first line of typed text sat
**23.0 px above** the centre of the `+` and Send buttons. Screenshot:
`before/before-light-1280-single.png`.
Root cause: three independent sources of truth. The buttons were hardcoded `height: 40px`; the
textarea was sized by `rows={2}` (two 22 px lines + 18 px padding + 2 px borders = 64 px). Its
`min-height: 40px` never applied, because `rows` already made it taller.
Fix: one `--composer-control-h: 40px` token in `:root` drives all three. The textarea's
single-line height is built from the same token — `--composer-field-pad-block: 8px` twice plus
`--composer-field-line-h: 22px` plus 2 px of borders is exactly 40 px — so a single line is
optically centred against the buttons arithmetically, with no pixel nudge. `box-sizing:
border-box` already comes from the global reset, so nothing was re-declared; there were no
`margin-top` / `position: relative` nudges to delete.
Verification (`composer-shots.mjs`, `optical.mjs`; headless Chrome, 5 widths × 2 themes × 4
states — 320/375/768/1280/1920, light/dark, empty/single/four-line-wrapped/past-the-cap):
first-line-centre offset from the buttons' centre **−23.0 px → 0.0 px** at both 375 px and
1280 px; textarea height 64 px → 40 px at a single line. All 40 rows keep
`bottom-spread = 0.00`, and the applied `data-theme` is asserted per row so the light and dark
passes cannot silently be the same run.
Correction to an earlier draft of this entry: `bottom-spread = 0.00` held *before* the change too
— the row was already bottom-aligned. It is a no-regression check here, not evidence of the fix.
The fix is evidenced by the offset and height figures above.
Regression check: `+` still opens the file picker; Enter with an attachment still submits (chat
turns 0 → 2) and inserts no newline; Shift+Enter still inserts a newline; Tab order is
`+` → textarea → Send.

[UI] [FRONTEND] Composer textarea did not grow with its content (2026-09-20)
Scope: as above.
Symptom: the textarea was a fixed two-row box at every length of input. With 80 lines of text in
it, it was still 64 px tall and the content was clipped at the bottom border with no way to see it
while typing. `max-height: 160px` in the stylesheet was dead code, because nothing ever set a
height for it to cap. `resize: none` was already set, so there was no manual workaround either.
Root cause: `rows={2}` and no measurement code.
Fix: `rows={1}` plus a `useLayoutEffect` that sets the height from `scrollHeight`. `scrollHeight`
excludes borders under `box-sizing: border-box`, so the measured border width
(`offsetHeight - clientHeight`) is added back to keep the border-box height exact. CSS caps the
field at `--composer-field-max-h: 160px` with `overflow-y: auto` past the cap.
Verification: measured across the full matrix — empty 40 px; four wrapped lines 106 px
(40 + 3 × 22); past the cap 160 px at **every** width and theme, with the content genuinely
overflowing (7 lines at 1920 px up to 80 lines at 320 px). `bottom-spread` stays 0.00 at the cap
boundary, so the buttons do not drift when the field stops growing. Live keyboard check:
Shift+Enter grew the field 40 px → 62 px.
Regression check: send, attach and Enter-to-send paths re-asserted (previous entry).

[UI] [FRONTEND] Auto-grown height went stale when the field changed width (2026-09-20)
Scope: as above.
Symptom: found while verifying the fix above. The same text wraps to a different number of lines
as the field narrows, but the height was recomputed on input only, so it was wrong after a resize
or a rotation. Measured at 320 px on first paint: an empty composer rendered 84 px tall, because
the height had been measured before the layout settled and was never revisited.
Root cause: the measurement ran on `[prompt]` only.
Fix: a `ResizeObserver` on the textarea re-fits the height when its **width** changes. The
observed width is compared explicitly against the previous one, so re-fitting the height cannot
re-trigger the observer.
Second-order finding, fixed in the same pass: Chrome counts the *placeholder* in a textarea's
`scrollHeight`. At 320 px the placeholder wraps to two lines while the value is empty, so an empty
field measured 62 px. Suppressing that was tried and produced a worse result — the placeholder was
then clipped mid-line against the bottom border. The field is now sized to whatever it is actually
showing, placeholder included: two lines of placeholder give a two-line-tall empty field on a
phone, exactly as two typed lines would, and the row's bottom edge still aligns. This also absorbs
Phase 2's longer Indic placeholders with no further change.
Verification: `after/after-light-320-empty.png` — placeholder fully legible, no clipping, three
controls on one baseline. Matrix re-run in full after the change; all 40 rows
`bottom-spread = 0.00`.

[UI] [FRONTEND] One focus ring for the composer row (2026-09-20)
Scope: as above.
Symptom: `.chat-composer__input:focus` set `outline: none` and changed only the border colour, and
the two buttons had no focus style at all — keyboard focus on `+` and Send was invisible.
Fix: `--focus-ring-w` / `--focus-ring-offset` tokens applied as one `:focus-visible` rule shared by
`+`, the textarea, Send, and the in-field icon button Phase 4 will add. The textarea keeps its
border-colour change for pointer focus. `outline: none` was removed, not replaced with a second
suppression.
Verification (`focus-order.mjs`): Tab into each control in turn and read the computed outline of
all three — `focused=attach → [attach]`, `focused=textarea → [textarea]`, `focused=send → [send]`,
exactly one ring at each stop, never two. Screenshots: `focus/`.
Note: Send is `disabled` until a file is attached, and a disabled button is correctly skipped by
Tab; the tab-order check attaches a real file first so all three controls are reachable.

[UI] [FRONTEND] In-field icon row established for Phase 4 (2026-09-20)
Scope: as above.
Note, not a fix: Phase 1 §6 of the work order describes aligning "the existing assistant icon"
inside the textarea. There is no in-field icon in this composer — the only icons in
`frontend/public/icons.svg` are unused template leftovers (bluesky, discord, x). Rather than invent
a decorative icon that does nothing, the row's geometry was built and left empty.
What exists now: `.chat-composer__icons` is anchored to the field's bottom-right with one
`--composer-icon-size`, one `--composer-icon-gap` and one `--composer-icon-inset`, and the
textarea's `padding-inline-end` is derived from the icon count via `:has()`, so text can never run
under an icon. Adding the mic in Phase 4 needs no stylesheet change: the component renders the row
with `--composer-icon-count: 1` and the reserved padding follows. Because the row sits inside the
field after the textarea, the mic lands in tab order between the textarea and Send, which is the
order the work order asks for.

[FINDING] [FRONTEND] Chat header forces horizontal page scroll at ≤400 px (2026-09-20)
Not fixed — outside this work order's scope; recorded in KNOWN_GAPS.md §8.
Symptom: `/assistant` scrolls horizontally on a phone. Measured `document.scrollWidth` 401 px
against both a 320 px and a 375 px viewport.
Root cause: `ModelSelector`'s native `<select>` renders at its widest option ("More controllers —
coming soon (not available)") — 329 px, inside a 400 px label that does not shrink. The composer
itself fits: at 320 px its row is 288 px wide and does not overflow.
Why it is recorded here: the composer's 320 px behaviour was verified against this pre-existing
page overflow rather than in spite of it — the field is ~169 px wide at 320 px either way.

[FEATURE] [FRONTEND] Interface multilinguality — English, Hindi, Kannada, Telugu, Tamil (2026-09-20)
Scope: Phase 2 of the work order — interface strings only. The query/answer language path is
Phase 3 and was not started: the string POSTed to /api/query is still English in every locale.
Evidence: `/Users/sukesh/Desktop/satquery-phase2-evidence/`.

Infrastructure: `frontend/src/i18n/` — `en.ts` (source of truth), `hi.ts`, `kn.ts`, `te.ts`,
`ta.ts`, `types.ts`, `I18nProvider.tsx`, `useT.ts`. 131 flat semantic keys per locale. No new npm
dependency; no i18next, no react-intl. Locale choice persists in `localStorage` under
`satquery.lang` (added to the existing `STORAGE_KEYS`), resolving stored → `navigator.languages`
primary subtag → `en`. `document.documentElement.lang` is set on change. Interpolation is a
`{name}` substitution used by nine keys; no plural engine was built because no string needed one
(the 1-vs-2 image count is two separate keys, since the composer caps attachments at two).
Key parity is a compile error, not a runtime fallback: `Locale = Record<TranslationKey, string>`
where `TranslationKey = keyof typeof en`.
Verified both directions by breaking it on purpose —
  removed `composer.send` from kn.ts → `src/i18n/kn.ts(4,14): error TS2741: Property
  '"composer.send"' is missing in type '{...}' but required in type 'Locale'.`
  added a typo'd `composer.sendd` → `src/i18n/kn.ts(54,3): error TS2353: Object literal may only
  specify known properties, and ''composer.sendd'' does not exist in type 'Locale'.`
Regression check: `tsc -b` clean and `npm run build` clean after each conversion step.

Runtime parity test: `frontend/scripts/check-locale-parity.mjs`, run with `npm run check:i18n`.
This repo has no test runner (no vitest/jest; `"lint": "oxlint"` is the only check), and the work
order forbids adding a dependency, so the script uses the esbuild that Vite already depends on to
load the five TypeScript catalogs and compare them. It also reports values left identical to
English, and keys defined but never referenced from `src/` — neither of which the type system can
express. Output: `en.ts defines 131 keys / hi,kn,te,ta 131 keys PARITY OK / Every key is
referenced from src/ / All locales have identical key sets`, exit 0.
Failure mode verified: deleting a key and un-translating another produced
`te.ts 130 keys PARITY FAILED / missing (1): composer.send / NOTE — 1 value(s) identical to
English: nav.home`, exit 1. The unused-key check caught four dead keys on its first run
(`home.stepMore`, `trace.none`, `trace.title`, `upload.removeFile`), which were removed.

[FEATURE] [FRONTEND] Per-script webfont loading (2026-09-20)
Scope: `I18nProvider.tsx` (injection), `i18n/types.ts` (the per-locale font table), `index.css`
(four `html[lang=…]` rules).
Problem: IBM Plex Sans carries no Indic glyphs, so four of the five languages would render as
tofu. Checked against Google Fonts directly: IBM Plex has a Devanagari cut, but **IBM Plex Sans
Tamil and IBM Plex Sans Kannada do not exist** — those requests return a 404 HTML page, and the
family list stops at Arabic, Devanagari, Hebrew, JP, KR and Thai. Hindi therefore keeps the app's
own type voice (IBM Plex Sans Devanagari); Kannada, Telugu and Tamil use Noto Sans, which is the
closest match in tone and is designed to harmonise across scripts.
Implementation: the provider appends one `<link rel="stylesheet">` for the selected locale's
script, once, keyed by element id; `html[lang="…"]` puts that family in front of the existing
Latin stack in `--font-sans`. `--font-mono` is deliberately untouched, so coordinates, GSD,
confidence values, CRS strings and trace keys stay in IBM Plex Mono and in Latin script.
Verification (headless Chrome, CDP Network domain, per locale):
  en  lang=en  injected <link>: []              Indic font requests: 0 (none)
  hi  lang=hi  injected: [satquery-font-hi]     5 requests → IBM+Plex+Sans+Devanagari
  kn  lang=kn  injected: [satquery-font-kn]     5 requests → Noto+Sans+Kannada
  te  lang=te  injected: [satquery-font-te]     5 requests → Noto+Sans+Telugu
  ta  lang=ta  injected: [satquery-font-ta]     5 requests → Noto+Sans+Tamil
English loads no Indic font, and no locale loads a script it does not use. Measured payload per
family (400+600, full subset): Devanagari 156 KB, Kannada 129 KB, Telugu 97 KB, Tamil 97 KB —
against 197 KB for the app's existing Latin faces. Loading all four eagerly would have roughly
tripled font weight for every visitor, including English ones.
Regression check: `viewer`/`confidence` values re-checked in the Tamil screenshot — `0.97` and
`R1`–`R5` still render in IBM Plex Mono, in Latin.

[FEATURE] [FRONTEND] Language switcher — custom listbox in the nav (2026-09-20)
Scope: new `frontend/src/components/LanguageSwitcher.tsx`; `Navbar.tsx` renders it inside
`.navbar__controls` (not `.navbar__links`), so it stays reachable when the nav collapses to a
menu at ≤640 px. Styles use existing tokens only.
No native `<select>`, per the work order. Each language is named in its own script — English,
हिन्दी, ಕನ್ನಡ, తెలుగు, தமிழ் — never a flag or country code.
Verification (headless Chrome, keyboard events dispatched through CDP):
  closed        aria-haspopup=listbox, aria-expanded=false
  ArrowDown     open=true role=listbox options=[option,option,option,option,option]
                aria-selected count=1  active="English"  focus moved into the list
  ArrowDown     active="हिन्दी"      End  active="தமிழ்"      Home  active="English"
  Escape        open=false, focus returned to trigger = true
  Space→↓↓→Enter  highlighted "ಕನ್ನಡ" → lang=kn, trigger=ಕನ್ನಡ, stored=kn,
                focus returned to trigger = true
  click outside closes: true      scroll closes: true
Switching does not reload and does not lose work in progress: with text typed into the composer,
selecting Kannada gave `NO PAGE RELOAD: true`, the typed string preserved verbatim, and the Send
button relabelled live to "ಕಳುಹಿಸಿ".
Motion: one `var(--transition-fast)` open transition, disabled under `prefers-reduced-motion`.
Known behaviour, deliberate: option labels for scripts whose webfont is not currently loaded fall
back to the operating system's font for that script. Preloading all four Indic families just to
draw this menu would defeat the per-script loading above and make an English visitor download
fonts they never read text in.

[UI] [FRONTEND] Narrow-screen horizontal overflow fixed — closes KNOWN_GAPS §8 (2026-09-20)
Scope: `index.css` — `.model-selector`, `.model-selector__select`, and `.chat-header .pill`
inside the existing ≤900 px media query.
Symptom: `/assistant` scrolled horizontally on a phone. Recorded in Phase 1 as KNOWN_GAPS §8 at
401 px against a 320 px viewport. The Phase 2 length stress test made it worse and showed a second
cause: 445 px in English and **531 px in Tamil**, because the translated controller label is wider
than "Controller", and because the backend capability pill ("LLM: qwen3:4b (ollama, Q4_K_M) · …")
only renders when a backend is reachable — it was absent from the Phase 1 measurement, which was
taken with the backend down.
Root cause: a flex item's `min-width` defaults to `auto`, so the native `<select>` — which sizes
itself to its widest option — could not shrink; and `.pill` is `white-space: nowrap`.
Fix: `min-width: 0; max-width: 100%` on the selector and its `<select>`, and the capability pill
wraps instead of widening the page on narrow screens. No `text-overflow: ellipsis` was applied to
any prose: the select truncates its own option text natively and the full text stays in the
control's `title`.
Verification, 320 px, all five languages × both themes, after the fix:
  en/hi/kn/te/ta, light and dark — `page 320px vs viewport 320px  no hscroll  | clipped: none`
(10/10). The same probe also checks nav links, the switcher trigger, evidence chips, sample
titles, rail labels, toggles, export links and the confidence label for `scrollWidth >
clientWidth`; nothing clips in any language, so no string needed truncating.
Regression check: Phase 1's composer matrix re-run after these stylesheet edits — 40/40 rows still
`bottom-spread = 0.00`.

[FEATURE] [BACKEND] Query and answer language path — the translation seam (2026-09-20)
Scope: Phase 3. New `backend/qwen/translation.py`; the seam is called from exactly one place in
`job_manager.py` (STAGE 3a, immediately before MC2); `main.py` gains the API field;
`mc8_export/exporter.py` keeps exports English. No MC1–MC8 logic changed, and no translation
model, service or dependency was added — translation uses the Qwen3 instance already in the
pipeline.
Module named for the gate: **`backend/qwen/translation.py`**.

API: `POST /api/query` accepts `query_language` ∈ {en, hi, kn, te, ta}, default `"en"`; anything
else is HTTP 400. Documented in the README API table.
Verified over real HTTP that existing clients are unaffected — a request with no `query_language`
field at all:
  HTTP 200  job accepted: True
  status=DONE in 56s
  query_language recorded: 'en'
  input_translation present: False
  answer: Between the two images, there were 13 regions of change covering 3.15% of the scene…
and `query_language=de` → `HTTP 400  detail: Unsupported query_language 'de'; supported: en, hi,
kn, te, ta.`

Seam: if `query_language != "en"`, the query is translated to English and `query` is rebound
before `QueryIntelligencePipeline` is constructed, so MC2, MC3 and every engine see the English
string. The transformation is written to `job["input_translation"]`, to the result, and as a
`progress_trace` entry.
Regression check: `test_english_query_never_enters_the_translation_seam` asserts
`translate_query_to_english` is never called for an English job and that no `input_translation`
key appears; `test_translation_happens_before_mc2_and_is_recorded_in_the_trace` subclasses
`QueryIntelligencePipeline` to capture what MC2 actually received and asserts it is the English
string, not the Tamil one.

Honest degradation: with no language model, a non-English query is refused rather than guessed.
Verified against a second backend started with `SATQUERY_QWEN_BACKEND=none`:
  Kannada query → status=TRANSLATION_UNAVAILABLE
    execution_status : TRANSLATION_UNAVAILABLE
    query_language   : 'kn'
    claims made      : []
    confidence       : 0.0
    refusal message  : This deployment is running Tool Registry rules; non-English queries need
                       the language model. Ask in English, or connect a deployment that runs Qwen3.
  English query on the SAME deployment → status=DONE, answered by registry rules
    planner: {"query_intelligence": "registry_rules", "tool_selection": "registry_rules",
              "answer": "template"}
`TRANSLATION_UNAVAILABLE` was added to `TERMINAL_STATUSES`; without that the frontend would have
polled a finished job forever.
Regression check: `test_non_english_query_is_refused_when_no_language_model_can_translate_it`
patches `RegistryQueryInterpreter` and asserts it was never constructed — the Kannada string never
reaches English keyword matching.

Exports: `mc8_export/exporter.py` reads `final_answer_en` / `caveats_en` through a new
`_audit_text()` helper, so the PDF is written from English even when the answer was shown in
another language, and the result carries `exports_language: "en"`. ReportLab's built-in Type 1
faces have no Indic glyphs, so the alternative was a PDF of empty boxes (KNOWN_GAPS §14).

[BUG] [BACKEND] Translation always failed: the engine reports success as "ok", not "success" (2026-09-20)
Symptom: every live non-English query terminated as `TRANSLATION_UNAVAILABLE` with "the language
model did not return a translation: unknown error", on a deployment where Qwen3 was reachable and
answering normally.
Root cause: `qwen/translation.py` treated `status == "success"` as success. Both engines in
`qwen/` return `"ok"` (`inference.py:134`, `ollama_inference.py`), so every translation was read
as a failure.
Why the tests did not catch it: the test stub returned `"success"` — it encoded the same wrong
assumption as the code, so ten passing tests sat on top of a feature that had never worked once.
The live end-to-end run is what exposed it.
Fix: check for `"ok"`, the convention both engines use.
Regression test added: `test_query_translation_refuses_an_unrecognised_status` feeds a stub that
returns `"success"` and asserts `TranslationUnavailable` is raised, so the contract is pinned in
the direction that failed; the shared stub now returns `"ok"` like the real engines.

[BUG] [BACKEND] Ollama engine read a temperature kwarg and discarded it (2026-09-20)
Symptom: caveats came back untranslated on a successful Hindi run — the Hindi answer was correct,
but `caveats[0]` was byte-identical to `caveats_en[0]`.
Root cause, found by replaying the exact prompt against the live model: at temperature 0.0 Qwen3-4B
fell into a repetition loop, emitting "अधिकतम अवधि के लिए" several hundred times until it hit the
token limit, leaving the JSON unterminated:
  status: ok
  {"translations": ["अक्सर अधिकतम अवधि के लिए अधिकतम अवधि के लिए अधिकतम अवधि के लिए …
  JSON parse failed: Unterminated string starting at: line 3 column 5 (char 26)
The guard in `translate_texts_from_english` did its job and returned the English list unchanged —
a mis-aligned caveat would attach the wrong limitation to the wrong claim — so this degraded
rather than corrupted. But the underlying cause was that `ollama_inference.generate()` computed
`temperature = kwargs.get("temperature", 0.7)` and then sent a hardcoded `"temperature": 0.0`,
so the temperature argument had no effect from any caller. (`qwen/inference.py`, the transformers
engine, honours it correctly; only the Ollama path was affected.)
Fix: `ollama_inference.generate()` now sends the temperature it was given and accepts
`repeat_penalty`. The default is **0.0**, which is exactly what it always sent, so every existing
caller — MC2 extraction, decomposition, tool selection, answer synthesis — is byte-for-byte
unchanged.

Correction, from measuring instead of assuming: **the temperature was not the cause and the
"fix" was a regression, so it was reverted.** Replaying the same three real strings at both
settings:
  ════ temperature 0.0 (original) ════
    QUERY   ok  {"translation": "What has changed between these two images?"}
    CAVEAT1 PARSED: प्राप्त करने की तारीखें अनुपस्थित; मान्य अपलोड क्रम: image_1 = पहले, image_2 = बाद।
    CAVEAT2 PARSED: रेडियो/ऑप्टिकल सिग्नल में परिवर्तन के स्थान को पहचानता है, लेकिन यह बदले हुए…
  ════ temperature 0.2 + repeat_penalty 1.15 ════
    QUERY   ok  {"task": "Translate the given Hindi sentence…", "rules_summary": [...]}   ← ignored the schema
    CAVEAT1 PARSED: (empty)
    CAVEAT2 PARSED: (empty)
At 0.0 all three translate correctly; at 0.2 the model answered the *instructions* instead of the
question and returned empty caveats. `qwen/translation.py` therefore uses `temperature=0.0`, and
the real cause of the repetition loop was batching several passages into one prompt — fixed by
translating one passage per call (next entry). The `ollama_inference` plumbing fix is kept because
the discarded-kwarg bug is real, but with the 0.0 default it is currently behaviour-neutral: no
caller passes a temperature today.

[BUG] [BACKEND] Batched caveat translation fabricated caveats (2026-09-20)
Symptom: on a real Hindi run the answer translated correctly but every caveat came back in
English, byte-identical to `caveats_en`.
Root cause: `translate_texts_from_english` sent all caveats as one numbered list. Qwen3-4B does not
attend to the individual items in that format. Two distinct failures were observed live: at
temperature 0.0 it looped one phrase ("अधिकतम अवधि के लिए" several hundred times) until the token
budget ran out and the JSON was unterminated; with a temperature and a repetition penalty it
returned the **same invented sentence for every passage** — which passes a count check, so it would
have shipped fabricated caveats attached to a real answer.
Why this mattered more than the latency it saved: caveats are what keep an answer honest. A caveat
that reads fluently and says the wrong thing is worse than one left in English.
Fix: one generation per passage, which is the path that demonstrably works, and which degrades per
passage — anything the model cannot translate comes back in English rather than as something
plausible and wrong. Cost: a non-English job makes 1 + 1 + N model calls, where N is the number of
caveats.
Regression tests added: `test_caveats_are_translated_one_passage_per_call` asserts two passages
produce two generations; `test_a_caveat_the_model_cannot_translate_stays_english_on_its_own`
asserts one failure leaves the other caveat translated rather than reverting the whole list.

[VERIFICATION] [BACKEND] Phase 3 end-to-end, five languages, real imagery (2026-09-20)
Every run below is the real MC1→MC8 pipeline on the real S1-AAD pair shipped with the site
(`s1aad_ID32_before.tif` / `_after.tif`), against a backend with Qwen3 reachable. No mocks, no
fixtures, no recorded answers.

English (no `query_language` field sent at all — the existing-client path):
  HTTP 200, status=DONE in 56s, query_language recorded 'en', input_translation absent.

Four non-English runs, all DONE:
  hi   97s   kn   92s   te  117s   ta  193s
Each recorded `input_translation` in the result and exactly one `progress_trace` entry carrying
it, kept `final_answer_en`, and reported `exports_language='en'`.

Trace excerpt, Kannada run (the artefact the work order asks for):
  {
    "message": "Query translated kn → en; the pipeline reasons over the English string shown here",
    "input_translation": {
      "from": "kn",
      "to": "en",
      "engine": "qwen3:4b",
      "original": "ಈ ಎರಡು ಚಿತ್ರಗಳ ನಡುವೆ ಏನು ಬದಲಾಯಿತು?",
      "translated": "What has changed between these two images?"
    },
    "timestamp": "2026-09-20T17:24:15.912940+00:00",
    "stage": "QUERY_INTELLIGENCE"
  }

Answers came back in the asked language with the figures intact — Kannada:
  "13 ಪ್ರದೇಶಗಳು … 3.15% ಪ್ರದೇಶವನ್ನು (126,100 ಮೀ²) … 5.80 ಡಿಬಿ …"
against the English original kept for audit:
  "Between the two images, there are 13 regions of change covering 3.15% of the scene (126,100 …"

Caveat translation, measured rather than claimed — 8 passages across the four languages:
  hi 2/2 translated | te 2/2 | ta 1/2 | kn 0/2   → 5 translated, 3 returned in English
Identifiers survived where translation succeeded (`image_1 = पहले, image_2 = बाद`). The three
failures returned English rather than inventing text, which is the intended degradation. Recorded
in KNOWN_GAPS §16 with the per-language rate, because a Kannada user currently reads a Kannada
answer with English caveats.

Latency: a non-English job costs 1 (query) + 1 (answer) + N (caveats) extra model calls, which is
most of the difference between 56s for English and 92–193s for the others on this machine.

[VERIFICATION] [FRONTEND] Phase 3 through the browser, not the API (2026-09-20)
A real Kannada query typed into the UI, with the real sample pair attached, against the running
backend. Driven in headless Chrome; the job it created was then read back from the API.

  UI language: kn | switcher: ಕನ್ನಡ
  answered at t+304s
  query_language reaching the backend : 'kn'
  input_translation recorded          : True
      ಈ ಎರಡು ಚಿತ್ರಗಳ ನಡುವೆ ಏನು ಬದಲಾಯಿತು? -> What has changed between these two images?
  exports_language                    : 'en'
  final_answer_en kept                : True

The trace's new first section renders in the interface language and shows both strings:
  translationSection: "00ಪ್ರಶ್ನೆಯ ಅನುವಾದ"
  translationKv: [["ಕೇಳಿದಂತೆ", "ಈ ಎರಡು ಚಿತ್ರಗಳ ನಡುವೆ ಏನು ಬದಲಾಯಿತು?"],
                  ["ವಿಶ್ಲೇಷಿಸಿದಂತೆ (ಇಂಗ್ಲಿಷ್)", "What has changed between these two images?"],
                  ["engine", "qwen3:4b"]]
  englishNote: "execution trace ಮತ್ತು ರಫ್ತು ಮಾಡಿದ ವರದಿ ಆಡಿಟ್ ದಾಖಲೆಗಳು, ಅವು ಇಂಗ್ಲಿಷ್‌ನಲ್ಲಿಯೇ ಉಳಿಯ…"

Screenshot `satquery-phase3-evidence/kn-answer-with-trace.png` shows the whole turn: the answer in
Kannada with every figure intact (13, 3.15%, 126,100 ಮೀ², 52,500 / 22,900 / 15,300 / 9,200 /
4,400), the confidence label in Kannada with `0.97` still in IBM Plex Mono, evidence chips reading
"Classical change detection · ಪ್ರದೇಶ R1" (engine name and region id kept in Latin), and the
viewer, checkbox and composer chrome in Kannada.
It also shows the limitation from KNOWN_GAPS §16 plainly: the caveats heading is Kannada
("ಎಚ್ಚರಿಕೆಗಳು") while both caveats are still English — the Kannada 0/2 result, visible rather than
hidden.

Note on the check that did not work: the script asserted `query_language` appeared in the POST
body and reported `false`. That is a limitation of the probe, not of the code — CDP does not expose
multipart/form-data request bodies. The field is proven to have arrived by the backend recording
`query_language: 'kn'` and producing `input_translation`, which only happens for a non-English
value.
Latency: 304s through the UI against 92s for the same Kannada query over the API earlier — the
difference is the per-caveat translation attempts, two of which failed and were retried before
falling back to English.
