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
