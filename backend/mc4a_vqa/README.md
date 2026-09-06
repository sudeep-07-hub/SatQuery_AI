# MC4A: Single-Image Perception Engine (PaliGemma VQA)

This module implements the MC4A single-image perception engine, initially providing visual question-answering (VQA) capabilities via the PaliGemma-3B model.

## Gated Model Access

The primary model `google/paligemma-3b-pt-224` is a gated repository on the Hugging Face Hub. To run this module:

1. **Accept the License:** You must accept the model's license agreement on its Hugging Face repository page: [https://huggingface.co/google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224).
2. **Environment Variable:** You must set the `HUGGING_FACE_HUB_TOKEN` environment variable with a valid access token that has permission to access the gated model.

```bash
export HUGGING_FACE_HUB_TOKEN="hf_..."
```
