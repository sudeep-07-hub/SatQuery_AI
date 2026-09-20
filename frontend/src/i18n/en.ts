/**
 * Source of truth for every interface string.
 *
 * Keys are flat, namespaced and semantic. Identifiers never appear here: MC1–MC8 stage codes,
 * tool and model names (PaliGemma, ChangeMamba, CROMA, Qwen3), status enums (ABSTAIN,
 * INSUFFICIENT_OBSERVATIONS), CRS strings, file extensions, API field names and execution-trace
 * keys stay in Latin script and untranslated, because they are the auditable values. Where a
 * status enum is shown, these strings are the human gloss printed *beside* the enum, not instead
 * of it.
 */
export const en = {
  // ── Navigation ──
  'nav.home': 'Home',
  'nav.assistant': 'Assistant',
  'nav.github': 'GitHub',
  'nav.githubAria': 'SatQuery AI on GitHub',
  'nav.brandAria': 'SatQuery home',
  'nav.openMenu': 'Open menu',
  'nav.closeMenu': 'Close menu',
  'nav.toggleTheme': 'Toggle colour theme',
  'nav.language': 'Language',
  'nav.languageChoose': 'Choose interface language',

  // ── Home ──
  'home.tagline': 'Ask questions of satellite imagery. Get answers tied to evidence.',
  'home.pipelineTitle': 'How a query runs',
  'home.pipelineAria': 'How a query is processed',
  'home.launch': 'Launch Assistant',

  // ── Pipeline steps (the real MC1→MC8 sequence) ──
  'pipeline.qualify.label': 'Input qualification',
  'pipeline.qualify.detail': 'Reads the uploaded files and checks whether they can answer the question.',
  'pipeline.qualify.more': 'Format, coordinate system, footprint overlap and sensor type. A pair with no overlap is rejected here.',
  'pipeline.query.label': 'Query understanding',
  'pipeline.query.detail': 'Turns the question into a task the pipeline can plan for.',
  'pipeline.query.more': 'Qwen3 4B (Q4_K_M, served by Ollama). If it is unavailable, deterministic Tool Registry rules take over and the answer says so.',
  'pipeline.registry.label': 'Tool registry',
  'pipeline.registry.detail': 'Picks an engine that can actually run in this environment.',
  'pipeline.registry.more': 'Each engine reports availability per job, so an engine that cannot run is never selected — ChangeMamba, for example, needs CUDA.',
  'pipeline.models.label': 'Specialist models',
  'pipeline.models.detail': 'Runs the selected engine on the uploaded pixels.',
  'pipeline.models.more': 'PaliGemma answers questions about a single image; change detection uses a classical SAR log-ratio / optical change vector analysis.',
  'pipeline.verify.label': 'Verification',
  'pipeline.verify.detail': 'Checks each piece of evidence before it reaches the answer.',
  'pipeline.verify.more': 'Evidence below the confidence threshold is rejected and listed as rejected. Confidence is raw model confidence, not calibrated.',
  'pipeline.answer.label': 'Answer + audit',
  'pipeline.answer.detail': 'Writes the answer from the verified evidence only.',
  'pipeline.answer.more': 'Each answer carries its evidence, an auditable execution trace, and PDF, GeoJSON and JSON exports.',

  // ── Sample queries ──
  'sample.airstrip.title': 'New airstrip in the Amazon',
  'sample.change.title': 'What changed here?',
  'sample.river.title': 'Ask about one image',
  'sample.lead': 'Sample queries — each one uploads the bundled imagery shown on the card and runs the pipeline live.',
  'sample.runs': 'Runs: {query}',
  'sample.oneImage': '1 image',
  'sample.twoImages': '2 images',
  'sample.unavailable': 'Not available on the connected backend',

  // ── Composer ──
  'composer.placeholder': 'Attach an image, then ask a question…',
  'composer.placeholderWithFiles': 'Ask a question about the attached image(s)…',
  'composer.attachAria': 'Attach images',
  'composer.attachTitle': 'Attach GeoTIFF, PNG or JPEG (or drop files here)',
  'composer.attachMax': 'Maximum {max} images',
  'composer.send': 'Send',
  'composer.analyzing': 'Analyzing…',
  'composer.dropHint': 'Drop imagery to attach',
  'composer.pairNote': 'sent together as one bi-temporal pair',
  'composer.removeFile': 'Remove {name}',

  // ── Chat thread ──
  'thread.imageA': 'Image A',
  'thread.imageB': 'Image B',
  'thread.image': 'Image',
  'thread.newConversation': 'New conversation',
  'thread.caveats': 'Caveats',
  'thread.imageryGone': 'The imagery for this answer is no longer held by the backend (jobs are kept in server memory), so the map cannot be redrawn.',

  // ── Status glosses (shown beside the verbatim enum, never instead of it) ──
  'stop.PRECONDITION_FAILED': 'Inputs rejected',
  'stop.INSUFFICIENT_OBSERVATIONS': 'Inputs do not fit the request',
  'stop.ABSTAIN': 'No answer given',
  'stop.INSUFFICIENT_EVIDENCE': 'Insufficient evidence',
  'stop.MODEL_UNAVAILABLE': 'Model unavailable',
  'stop.FAILED': 'Analysis failed',

  // ── Confidence ──
  'stop.TRANSLATION_UNAVAILABLE': 'Query language not supported here',
  'trace.inputTranslation': 'Query translation',
  'trace.translatedFrom': 'Translated from {lang}; the pipeline reasoned over the English text below.',
  'trace.originalQuery': 'As asked',
  'trace.englishQuery': 'As analysed (English)',
  'trace.englishNote': 'The execution trace and the exported report are audit artefacts and stay in English.',
  'confidence.uncalibrated': 'Model confidence (uncalibrated)',
  'confidence.calibrated': 'Calibrated confidence',
  'confidence.rawTitle': 'Raw model confidence — MC6.2 calibration has not run',

  // ── Evidence ──
  'evidence.aria': 'Evidence behind this answer',
  'evidence.rejected': 'rejected',
  'evidence.unknownEngine': 'Unknown engine',

  // ── Execution trace ──
  'trace.plannedTask': 'Planned task',
  'trace.noTask': 'The job stopped before a task was planned.',
  'trace.executedTools': 'Executed tools',
  'trace.noTools': 'No specialist tool was executed.',
  'trace.verification': 'Verification',
  'trace.registryFallback': 'Tool Registry rules (fallback)',

  // ── Sidebar ──
  'trace.show': 'Show execution trace',
  'trace.hide': 'Hide execution trace',
  'trace.evidenceRegions': 'Evidence regions',
  'trace.pipelineStages': 'Pipeline stages',
  'trace.noVerification': 'Verification did not run for this job.',
  'trace.noEvidence': 'No evidence objects were produced.',
  'trace.recoveries': 'Recoveries: {count}',
  'trace.noSpatialRegion': 'no spatial region',
  'trace.wholeImageFootprint': 'whole image footprint',
  'trace.localizedRegion': 'localized region',
  'trace.highlightOnMap': 'Highlight on the map',
  'trace.noRegionForEvidence': 'This evidence has no spatial region',
  'trace.rejectedPrefix': '[Rejected] ',
  'trace.modelConfidence': 'model confidence (uncalibrated)',
  'trace.plannerTemplate': 'evidence-only template (no LLM)',
  'trace.plannerFixture': 'test fixture',
  'sidebar.aria': 'Chat history',
  'sidebar.expand': 'Expand chat history',
  'sidebar.collapse': 'Collapse chat history',
  'sidebar.newChat': 'New chat',
  'sidebar.newChatTitle': 'Start a new chat',
  'sidebar.newChatBusy': 'Wait for the current response to finish',

  // ── Exports ──
  'sidebar.empty': 'No chats yet.',
  'sidebar.deleteChat': 'Delete chat',
  'sidebar.deleteChatAria': 'Delete chat “{title}”',
  'confidence.percentAria': '{pct} percent',
  'evidence.noRegion': 'no region',
  'evidence.wholeImage': 'whole image',
  'evidence.regionR': 'region R{n}',
  'export.pdf': 'PDF report',
  'export.geojson': 'GeoJSON',
  'export.json': 'JSON trace',

  // ── Uploads & errors ──
  'upload.maxFiles': 'Maximum {max} images allowed. Remove one before adding another.',
  'error.noImage': 'Attach at least one image to analyse.',
  'error.noQuestion': 'Enter a question about the attached image(s).',
  'error.unreachable': 'Could not reach the SatQuery backend, so your question was not analysed. Check that the server is running and try again.',
  'error.jobGone': 'The backend no longer has this job (it may have been restarted), so there is no result to show.',
  'error.serverFailed': 'The analysis failed on the server, so no answer was produced. The execution trace shows where it stopped.',
  'error.lostContact': 'Lost contact with the SatQuery backend while the analysis was running, so no answer was received.',

  // ── Viewer ──
  'error.endedWithStatus': 'The request ended with status {status}.',
  'error.sampleLoad': 'The bundled sample imagery could not be loaded. Attach your own images instead.',
  'sidebar.openDrawer': 'Open chat history',
  'sidebar.closeDrawer': 'Close chat history',
  'home.lead1': 'SatQuery AI is an agentic assistant for remote-sensing images: upload one or two optical or SAR images and ask a question in plain language.',
  'home.lead2': 'The controller interprets the query, selects specialist models from a tool registry, and returns an evidence-grounded response built only from what those models produced.',
  'home.lead3': 'Each answer includes its evidence items, an auditable execution trace, and model confidence reported as uncalibrated.',
  'viewer.before': 'Before',
  'viewer.after': 'After',
  'viewer.changeMaskToggle': 'Change mask',
  'viewer.evidenceRegions': 'Evidence regions',
  'viewer.map': 'Map',
  'viewer.swipe': 'Swipe',
  'viewer.notGeoreferenced': 'Not georeferenced: shown in image coordinates.',
  'viewer.changeMask': 'change mask',

  // ── Controller selector ──
  'controller.label': 'Controller',
  'controller.comingSoon': 'More controllers — coming soon (not available)',
  'controller.onlyOne': 'Only one controller exists today; the selection does not change any request.',
} as const;
