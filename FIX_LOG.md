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
family (400+600, full subset): **Hindi (IBM Plex Sans Devanagari) 142 KB over 8 files, Kannada
129 KB, Telugu 168 KB, Tamil 97 KB** — against 197 KB for the app's existing Latin faces. Loading
all four eagerly would have roughly tripled font weight for every visitor, including English ones.
Correction to an earlier draft of this entry, which quoted "Devanagari 156 KB … Telugu 97 KB":
156 KB is Noto Sans Devanagari, which is not the family used for Hindi, and the Telugu figure was
carried over from Tamil rather than measured. Re-measured at close-out; the numbers above are the
measured ones.
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

[FEATURE] [FRONTEND] Voice query input — browser-native dictation (2026-09-20)
Scope: Phase 4. New `frontend/src/lib/speech.ts` (recogniser + state machine) and
`frontend/src/components/assistant/MicButton.tsx`; `ChatComposer.tsx` wires them; `index.css` adds
the listening state, the dismissible error block and `.sr-only`; 14 new keys × 5 locales. No
backend endpoint was added — the Web Speech API runs in the browser, and a server-side Whisper
would not fit the 512 MB hosted deployment.
Evidence: `/Users/sukesh/Desktop/satquery-phase4-evidence/`.

Placement: the mic goes into the in-field icon row reserved in Phase 1 — same
`--composer-icon-size`, `--composer-icon-gap` and `--composer-icon-inset` as any other icon there,
and the component sets `--composer-icon-count: 1` so the textarea's reserved trailing padding
follows automatically. No stylesheet change was needed for the geometry, exactly as Phase 1
predicted.
Verified the reservation actually works with 220 characters of text:
  reservedPaddingRight 52px | text right edge 1124 | mic left edge 1143 | textStopsBeforeMic true

Phase 1 regression, re-run with the mic present:
  40/40 rows bottom-aligned; first-line offset from the buttons' centre 0.0px at 375px and 1280px.

Tab order is the order the work order asks for:
  .chat-composer__attach → #chat-prompt → .chat-composer__icon-btn → .chat-composer__send

Recognition language follows the interface language:
  en → en-IN | hi → hi-IN | kn → kn-IN | te → te-IN | ta → ta-IN

Dictation never sends. Interim results stream into the field and the final result lands editable:
  while listening: textarea "has a new airstrip", readOnly true, aria-pressed "true",
                   live region "Listening", ring animation "mic-ring"
  after final:     textarea "has a new airstrip been cleared?", editable again,
                   aria-pressed "false", **turnsAdded 0**
The field is read-only only while the recogniser is running, because an edit made mid-phrase would
be overwritten by the next transcript; it is editable again the moment dictation ends.

Every error code gets its own instruction — no generic "something went wrong":
  not-allowed         → Microphone access is blocked. Allow the microphone for this site…
  no-speech           → Nothing was heard. Check that the right microphone is selected…
  audio-capture       → No microphone was found. Connect one, then press the mic again.
  network             → Speech recognition runs as an online service and it could not be reached…
  aborted             → Dictation stopped before anything was recognised…
  service-not-allowed → The browser blocked the speech service for this page…
  weird-new-code      → Dictation stopped: weird-new-code. Press the mic to try again.
The last line is the point of the default branch: an unrecognised code still names what the
browser reported instead of hiding it.

Honest capability reporting, both kinds:
  no API at all (both constructors deleted before load) → mic stays visible, `disabled: true`,
    tooltip "Voice input needs a browser with speech recognition; this one does not have it."
  service rejects the language → mic disables for that language only, with the real reason in the
    interface language: "ಸ್ಪೀಚ್ ಸೇವೆ ಕನ್ನಡ ಅನ್ನು ಗುರುತಿಸುವುದಿಲ್ಲ…"
Denied permission does not re-prompt: the message persists and `start()` stayed at 1 call.

Limits: a 30-second cap is scheduled on start and calls `stop()` (not `abort()`, so a final result
still arrives) — `{"capScheduledMs":30000,"stopCalled":1}`. The cap is stated in the mic's tooltip
before the user starts rather than arriving as a surprise cut-off.

Reduced motion: `@media (prefers-reduced-motion: reduce)` replaces the pulse with a static filled
state — `ringAnimation "none", ringOpacity "1", background rgb(252, 231, 217)` (`--primary-tint`).

[BUG] [FRONTEND] Listening state was silently overridden by the shared icon-button rule (2026-09-20)
Symptom: under `prefers-reduced-motion: reduce` the mic's filled background did not render —
measured `backgroundColor: rgba(0, 0, 0, 0)` where `--primary-tint` was expected.
Root cause: `.mic--listening` (index.css:1195) and `.chat-composer__icon-btn` (index.css:1643) are
both single-class selectors, so they have equal specificity and the later one wins. The shared
icon-button rule sets `background: none` and `color: var(--ink-muted)`, so it was quietly undoing
the listening state — the colour and border change as well as the background.
Fix: scope the listening rules to `.chat-composer__icon-btn.mic--listening`, which wins on
specificity regardless of source order. **No `!important`** — the work order forbids it for
alignment, and it would have been the wrong tool here too.
Regression check: reduced-motion re-measured after the change →
`{"ringAnimation":"none","ringOpacity":"1","filledBackground":"rgb(252, 231, 217)","stillIndicatesListening":true}`

[CLOSE-OUT] [PHASE 5] Integration, documentation, bundle accounting (2026-09-20)

Alignment, re-run in every language with the mic present — the Phase 1 matrix × 5 locales:
  en 40/40 | hi 40/40 | kn 40/40 | te 40/40 | ta 40/40  →  **200/200 rows bottom-aligned**
(5 languages × 5 widths × 2 themes × 4 states: empty, single line, four wrapped lines, past the
cap). Screenshots: `satquery-phase5-evidence/matrix-<locale>/`.
Note on the first batch run: English reported 0 rows because the first Chrome of the batch had not
released its debugging port. Re-run on its own it gives 40/40, unchanged. A transient harness
failure, not a layout one.

Bundle, measured as bytes on disk and bytes gzipped — `0f631e9` (genuinely before this work order)
against the current tree:

  BEFORE  index-St4ykz3B.css   raw  60,101   gzip  14,853
          index-Cya13MNm.js    raw 429,629   gzip 133,472
  AFTER   index-B9YAy3X_.css   raw  65,373   gzip  15,783
          index-BRssdAh3.js    raw 521,137   gzip 153,748

  delta   CSS  +5,272 raw  (+930 gzip)      JS  +91,508 raw  (+20,276 gzip)
  total   **+21,206 bytes gzipped, +14.3%** over the whole four-phase work order.

Most of the JS growth is the five translation catalogs, which ship in the main chunk. Note the
figures above are **bytes**, not vite's reported "kB": vite counts characters, and Indic text is
three bytes per character in UTF-8, so vite under-reports this bundle by ~37 KB (484.24 kB
reported against 521,137 bytes on disk). Bytes are what crosses the wire, so bytes are quoted.

Fonts are *not* in those numbers — they load per script, on selection:
  English  0 KB (the Latin faces are already in index.html)
  Hindi    142 KB (IBM Plex Sans Devanagari, 8 files)
  Kannada  129 KB (Noto Sans Kannada, 3 files)
  Telugu   168 KB (Noto Sans Telugu, 3 files)
  Tamil     97 KB (Noto Sans Tamil, 3 files)
So an English visitor pays the +21 KB of catalogs and nothing else; the worst case is a Telugu
visitor at +21 KB + 168 KB. Eagerly loading all four families would have cost every visitor 536 KB
of font on top, which is why the provider injects one stylesheet on demand instead.
Deferred, not done: splitting the four non-English catalogs out of the main chunk would return
most of the +21 KB to English visitors. It needs dynamic `import()` and a loading state, and was
out of scope for a work order that forbids new dependencies and gates each phase.

Which caller gets what from the identifier policy, re-checked at close-out across the five
languages: model and tool names, `MC1`–`MC8`, status enums, CRS strings, file formats, API fields
and trace keys are Latin in every locale; `--font-mono` is untouched, so numbers, coordinates and
confidence values stay in IBM Plex Mono.

[UI] [FRONTEND] Language switcher did not read as a language control (2026-09-21)
Symptom: the first person shown the Phase 2 build looked for a way to change language and did not
find it. The trigger was the word "English ▾", which reads as a label rather than a control,
sitting between a GitHub mark and a sun/moon icon.
Root cause: the work order asks for the current language named in its own script and forbids
flags and country codes. That was followed, but it left the trigger with no visual cue that it is
a chooser at all.
Fix: a globe glyph before the language name (`GlobeGlyph` in `LanguageSwitcher.tsx`, stroked in
`--ink-muted`, `aria-hidden` so screen readers still hear the translated `nav.languageChoose`
label). A globe denotes "language", not a country, so it keeps to the rule against flags.
Verification: `{"hasGlobe":true,"ariaHidden":"true","label":"English▾","triggerWidth":97,
"stillInsideViewport":true}` in both light and dark. Screenshots:
`satquery-phase5-evidence/switcher-globe-{light,dark}.png`.
Regression check: trigger grew 77 px → 97 px; still inside the viewport, and the 320 px
no-horizontal-scroll result from Phase 2 is unaffected because the switcher sits in
`.navbar__controls`, which already had room.

[BUG] [BACKEND] Evaluation runner wrote reports to a cwd-relative path (2026-09-21)
Closes KNOWN_GAPS §18.
Root cause: `EvaluationRunner.__init__` defaulted `output_dir="backend/data/reports"`, correct only
when the process starts at the repository root. The test suite runs from `backend/`, so every
full run created a stray `backend/backend/data/reports/` that `.gitignore` did not cover.
Fix: the default is now `None`, resolved as `Path(__file__).resolve().parents[1] / "data" /
"reports"`. Callers that pass an explicit `output_dir` are unchanged.
Verification: imported as `backend.evaluation.runner` it resolves to
`/Users/sukesh/Desktop/satquery/backend/data/reports`; checked `.endswith("satquery/backend/data/reports")`
→ True. Also fixed in passing: the new signature used `Optional` without importing it, which
would have raised `NameError` on import — caught by importing the module for real rather than
only parsing it.

[CLEANUP] [FRONTEND] Removed seven unreferenced components (2026-09-21)
Closes KNOWN_GAPS §12. Done on request.
Removed: `components/results/ResultPanel.tsx`, `results/TracePanel.tsx`, `results/JobStatus.tsx`,
`components/QueryBox.tsx`, `ProfilePanel.tsx`, `TestHarness.tsx`, `AnalyzeButton.tsx` — leftovers
from the pre-chat single-page UI, holding roughly 35 untranslated strings.
Regression check: external references re-counted immediately before deletion, 0 for all seven.
After deletion `tsc -b` exits 0, `vite build` succeeds, and `npm run check:i18n` still reports
"Every key is referenced from src/ / All locales have identical key sets". `--border-light` and
the two `--shadow-sm` users were in these files; `--shadow-sm` stays defined for the language
popover.

[BUG] [ENV] The test count was not reproducible because two interpreters were in play (2026-09-21)
Corrects KNOWN_GAPS §17, whose first diagnosis was wrong.
Earlier finding, as recorded then: four CROMA modules fail at import with `No module named
'configilm'`, so "add configilm to requirements.txt". Checking before editing showed `configilm`
**is installed** — for a different interpreter:
  /opt/homebrew/bin/python3   v3.14.6   configilm: YES
  /usr/bin/python3            v3.9.6    configilm: no
The earlier suite runs had resolved `python3` to 3.9; an interactive shell resolves it to 3.14.
So the missing-module failures were a property of which interpreter ran, not of the checkout.
Fix: `configilm>=0.4.10` added to `backend/requirements.txt` so a clean checkout can install it —
it genuinely is a dependency of `mc4c/semantic_head.py`. Deliberately **not** added to
`requirements-render.txt`: CROMA is disabled on the 512 MB Render deployment via
`SATQUERY_DISABLED_TOOLS`, so it would be weight for a tool that never runs there.
Still open under §17: pinning the interpreter (a venv documented in the README, or pytest
configuration), so "the suite" means one thing.

[FINDING] [ENV] Full suite under Python 3.14 is OOM-killed on this machine (2026-09-21)
Run: `/opt/homebrew/bin/python3 -m pytest -q --continue-on-collection-errors`, output captured
straight to `satquery-phase5-evidence/pytest-py314.txt` (an earlier attempt piped through `tail`
lost the summary line when the process died).
  collected under 3.14: 756, collection errors: 0      (3.9: 4 collection errors)
  process exit: 137 (SIGKILL) after 723 of 756 results
  of those 723:  677 passed | 8 failed | 37 skipped | 0 errors
The 8 failures are all BigEarthNet-dataset tests — `test_task5_6_features.py` ×6,
`test_task5_7_text.py::test_caption_selection_and_join`,
`test_task5_8r_readiness.py::test_real_bigearthnet_visual_features` — the same data dependency
seen under 3.9, and not caused by this work.
Cause of the kill: the last test to complete was
`test_task7_2_croma.py::…::test_19_non_croma_regression`; the next file,
`test_task7_3_qwen3.py`, constructs `Qwen3Inference()` (transformers, ~8 GB bf16) while CROMA's
weights and Ollama's resident Qwen3 are already in memory, on a 16 GB machine. This is the same
failure mode that froze the machine outright earlier in this work.
Not re-run: the remaining 33 heavy tests were deliberately **not** retried unattended, because the
last time this machine ran out of memory it hung and needed a restart.

[VERIFICATION] [ENV] Full suite completed under Python 3.14 (2026-09-21)
The 34 tests the OOM kill had prevented from running were re-run one file per process under a
memory watchdog (`satquery-phase5-evidence/heavy-tests/run-heavy.sh`: kills the test if
system-wide free memory falls below 12%, and logs it). Ollama's resident Qwen3 was unloaded and the
dev server stopped first.
  test_task7_3_qwen3                     12 passed           min free 14%
  test_task7_4_paligemma                 KILLED BY WATCHDOG  min free 10%
  test_task7_5_full_system                7 passed           min free 86%
  test_task7_5r_multitool                 2 passed           min free 87%
  test_task8_2_2_qwen3_live_controller    2 passed           min free 86%
  test_task8_2_2r_qwen3_memory_safety     2 passed           min free 87%
  test_task8_4_evaluation                 3 passed           min free 87%
The watchdog did its job: PaliGemma pushed free memory to 10% and was killed; free memory then
dipped to 5% while the OS reclaimed it, and recovered to 87% within seconds. The runner was paused
before the next file in case it was also heavy — it was not (`full_system` runs on fixtures and had
already completed), so the remaining four were resumed without PaliGemma.

Combined with the first run's 722 non-heavy tests (677 passed, 8 failed, 37 skipped):
  **756 collected | 705 passed | 8 failed | 37 skipped | 6 not run**   (705+8+37+6 = 756)
versus Python 3.9: 696 passed, 11 failed, 37 skipped, 4 collection errors.

What 739 is: 756 − 17 new Phase 3 tests = 739, the suite's collected count before this work. The
work order's "≥739 green" treated it as a pass count; it never was one on this machine.
