# Notebook-faithful alignment demo

## Build
- Replace the placeholder home page with a minimal white-and-blue interactive app for SFT, RLHF, DPO, and side-by-side comparison.
- Re-run the notebook's actual synthetic-corpus, SFT, reward-model, RLHF, and DPO code at its documented seeds/settings to export small model checkpoints. The notebook contains executable code and saved result tables, but no serialized weights; label these as reproduced checkpoints, not the author's original saved models.
- Add browser-side inference using the same next-token model and notebook gold/proxy/independent rubric functions. Treat input as a known-vocabulary prefix, clearly explain this model limitation, and reject unsupported words rather than pretending it understands arbitrary prompts.
- Show only scores computed from each generated response, with method selection and a comparison view; add route-specific metadata and keep dependencies unchanged.

## Technical details
- Keep inference client-side and stateless; no login, database, server API, personal details, or external model service.
- Keep the existing TanStack Start route structure and semantic design tokens; add no image or extra routes.
- Validate that the reproduced model output and displayed metrics are computed from checkpoint-backed generations, then check the preview build and home page.
