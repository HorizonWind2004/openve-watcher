"""官方 Kiwi-Edit 评分提示词的逐字副本，仅供测试比对。

取自 https://github.com/showlab/Kiwi-Edit 的 `eval_openve_gemini.py`
（常量段与 prompt_type 字典，去掉了 import 与执行逻辑）。
这个文件是基准，不要修改它来让测试通过——上游变了才更新，并同步 prompts.py。
"""

# The prompts are from OpenVE-3M (https://arxiv.org/abs/2512.07826).
GLOBAL_STYLE = """
You are a data rater specializing in grading video style transfer edits. You will be given an input video, a reference style (image or video), and the styled result video. Your task is to evaluate the style transfer on a 5-point scale from three perspectives:

Instruction Compliance
1. Target style absent or clearly wrong.
2. Style shows in a few areas/frames only, or mixed with unrelated styles.
3. Key traits (palette, brushwork, texture) present but patchy or inconsistent across frames.
4. Style reproduced well across almost the whole video; only small local or brief temporal mismatches.
5. Full, faithful transfer: colour, texture, brushwork, and lighting all match the exemplar consistently over the entire duration and space of the video.

Consistency & Detail Fidelity
1. Major objects, layout, or overall motion lost/distorted; original scene barely recognisable.
2. Main subject recognisable, but its size, perspective, motion, or key parts are clearly wrong/missing.
3. Overall structure and motion correct; some local warping, minor omissions, or slight motion jerkiness.
4. Nearly all geometry and motion intact; only slight, non-distracting deformation.
5. All objects, spatial relations, and motion are perfectly kept; only stylistic, harmless distortion.

Visual Quality & Stability
1. Extreme flickering or “boiling” effects; the style is completely unstable frame-to-frame, making the video unwatchable.
2. Significant and distracting flickering or temporal inconsistency in style application.
3. Noticeable but tolerable flicker or texture “boiling”, especially during motion.
4. Largely stable with only minor, subtle flickering visible in areas of complex motion or fine texture.
5. Perfectly stable and temporally coherent; the style appears “stuck” to the scene with no flickering.

Note: The scores for Consistency & Detail Fidelity and Visual Quality & Stability should not be higher than the Instruction Compliance score.

Example Response Format
Brief reasoning: A short explanation of the scores based on the criteria above, no more than 30 words.
Instruction Compliance: A number from 1 to 5.
Consistency & Detail Fidelity: A number from 1 to 5.
Visual Quality & Stability: A number from 1 to 5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


BACKGROUND_CHANGE = """
You are a data rater specializing in grading video background editing. You will be given two videos (before and after editing) and the editing instruction. Your task is to evaluate the background change on a 5-point scale from three perspectives:

Instruction Compliance
1. No change, or background unrelated to prompt, or foreground also replaced/distorted.
2. Background partly replaced or wrong style/content; foreground noticeably altered.
3. Main background replaced but elements missing/extra, or faint spill onto subject edges.
4. Requested background fully present; foreground intact except minute artefacts or small prompt mismatch (e.g. colour tone).
5. Background exactly matches prompt (content, style, placement); all foreground pixels untouched.

Consistency & Detail Fidelity
1. Large tearing, posterisation, or significant temporal artifacts like flickering, jittering edges; edit area obvious at a glance.
2. Clear cut-out halos, colour-resolution gap, or obvious edge instability over time.
3. Blend acceptable but visible on closer look: slight edge blur, or minor temporal instability.
4. Nearly invisible seams; edges are stable across motion, textures aligned, only minor issues when zoomed in.
5. Indistinguishable composite: edges, textures, resolution and colour grading are perfectly continuous and stable throughout the video.

Visual Quality & Stability
1. Severe mismatch: wrong horizon, conflicting light, floating subject, or static background during camera movement.
2. Noticeable inconsistencies in light or scale; incorrect perspective shifts during motion.
3. Overall believable; small errors in shadow, perspective, or minor motion tracking flaws.
4. Lighting, scale, and depth well matched; background tracks convincingly with camera motion.
5. Physically flawless: coherent light, shadows, perspective, and depth throughout.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Consistency & Detail Fidelity: 1-5.
Visual Quality & Stability: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


LOCAL_CHANGE = """
You are a data rater specializing in grading video replacement edits. You will be given two videos (before and after editing) and the editing instructions.

Instruction Compliance
1. Target not replaced or unrelated edit.
2. Partial replacement or wrong class.
3. Largely replaced but with visible remnants or incorrect count/position.
4. Correct replacement with minor attribute errors.
5. Perfect replacement matching class, number, position, scale, pose, motion, and detail.

Consistency & Detail Fidelity
1. Video heavily broken or object flickers uncontrollably.
2. Obvious seams, colour mismatch, or unstable background.
3. Mostly correct but noticeable flicker or lighting inconsistency.
4. Nearly seamless; only tiny temporal artefacts.
5. Completely seamless and temporally stable.

Visual Quality & Stability
1. Severe tracking, lighting, or perspective errors.
2. Missing shadows, poor occlusion, or mismatched motion.
3. Mostly correct with minor inconsistencies.
4. Well-tracked with realistic interactions.
5. Physically flawless integration.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Consistency & Detail Fidelity: 1-5.
Visual Quality & Stability: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


LOCAL_REMOVE = """
You are a data rater specializing in grading video object removal editing.

Instruction Compliance
1. No edit or completely wrong.
2. Wrong object removed or partial removal.
3. Correct object removed with major errors or ghosting.
4. Correct object removed with minor fragments.
5. Perfect removal with everything else untouched.

Visual Quality & Stability
1. Severe artefacts or flickering.
2. Obvious erase marks or jitter.
3. Noticeable temporal inconsistency.
4. Minor edge issues only on close inspection.
5. Perfectly seamless and stable.

Consistency & Detail Fidelity
1. Background badly reconstructed or static.
2. Background shifts or jitters over time.
3. Mostly correct with small flaws.
4. Clean and stable reconstruction.
5. Background perfectly matches original motion and detail.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


LOCAL_ADD = """
You are a data rater specializing in grading video object addition editing.

Instruction Compliance
1. No edit or wrong object added.
2. Partial or wrong addition.
3. Correct object added with major attribute errors.
4. Correct object with minor inaccuracies.
5. Perfect addition with all attributes correct.

Visual Quality & Stability
1. Severe artefacts or flickering.
2. Obvious paste marks or jitter.
3. Noticeable lighting or colour mismatch.
4. Minor edge or temporal artefacts.
5. Perfectly seamless and stable.

Consistency & Detail Fidelity
1. Severe physical errors or occlusion issues.
2. Poor contact, occlusion, or motion.
3. Mostly correct with minor flaws.
4. Realistic shadows, reflections, and motion.
5. Perfect physical and temporal integration.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


SUBTITLES_EDIT = """
You are a data rater specializing in grading instruction-following subtitle edits.

Instruction Compliance
1. Wrong subtitle or no edit.
2. Right action but wrong content or partial edit.
3. Mostly correct with significant errors.
4. Correct with minor inaccuracies.
5. Perfect subtitle edit with zero unintended changes.

Visual Quality & Stability
1. Attributes completely wrong or unreadable.
2. Major deviation from requested attributes.
3. Acceptable but inconsistent placement or style.
4. Minor inaccuracies only.
5. Perfect attribute matching or professional default choice.

Consistency & Detail Fidelity
1. Major video corruption or subtitle damage.
2. Noticeable artifacts or unintended subtitle changes.
3. Minor unintended effects.
4. Almost perfect preservation.
5. Perfect isolation of the edit.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


CAMERA_MULTI_SHOT_EDIT = """
You are a data rater specializing in grading camera shot type alteration edits.

Instruction Compliance
1. Shot type unchanged or wrong.
2. Direction correct but degree wrong.
3. Generally correct but poorly framed.
4. Correct shot with minor framing issues.
5. Perfect shot type and framing.

Visual Quality & Stability
1. Severe distortion or glitches.
2. Distracting jitter or warping.
3. Minor visual flaws.
4. Very stable with tiny artefacts.
5. Perfectly stable and clear.

Consistency & Detail Fidelity
1. Completely different scene.
2. Major illogical changes.
3. Noticeable continuity errors.
4. Highly consistent with minor discrepancies.
5. Perfect consistency and continuity.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


CREATIVE_EDIT = """
You are a data rater specializing in grading instruction-following creative video edits.

Instruction Compliance
1. Instruction ignored.
2. Attempted but fundamentally failed.
3. Generally follows instruction with major deviations.
4. Successful with minor inaccuracies.
5. Perfect creative execution throughout.

Visual Quality & Stability
1. Unwatchable due to flicker or artefacts.
2. Obvious temporal inconsistency or seams.
3. Mostly stable with noticeable boiling.
4. Very stable with subtle artefacts.
5. Perfectly seamless and stable.

Consistency & Detail Fidelity
1. Severe physical inconsistencies.
2. Major lighting or motion errors.
3. Mostly believable with minor flaws.
4. Realistic interaction and preserved details.
5. Indistinguishable from real footage.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


import base64
import json, os
prompt_type = {
    'global_style': GLOBAL_STYLE,
    'local_change': LOCAL_CHANGE,
    'background_change': BACKGROUND_CHANGE,
    'local_remove': LOCAL_REMOVE,
    'local_add': LOCAL_ADD,
}
