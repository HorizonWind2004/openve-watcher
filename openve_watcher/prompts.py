"""OpenVE 打分用的分类别评分提示词。

所有进入 `prompt_type` 的提示词都逐字对齐官方 Kiwi-Edit
(https://github.com/showlab/Kiwi-Edit 的 `eval_openve_gemini.py`)，
由 `tests/test_prompts_alignment.py` 逐字节断言，基准副本在
`tests/data/kiwi_eval_prompts.py`。

不要在这里改写措辞。官方改了就重新抽取，并更新那份基准副本。
原始提示词来自 OpenVE-3M (https://arxiv.org/abs/2512.07826)。
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


COLOR_CHANGE = """
You are a data rater specializing in strict local video color edits. You will be given two videos (before and after editing) and the editing instruction.

Instruction Compliance
1. The requested local color change is absent, targets the wrong object, changes the whole scene, or makes an unrelated edit.
2. The target color changes only weakly, inconsistently, or to the wrong color; unrelated areas are also changed.
3. The target color is recognizable but incomplete, patchy, or mixed with unwanted material/shape/background changes.
4. The target color is correct with only minor hue, boundary, or temporal inaccuracies; unrelated content is nearly preserved.
5. Only the specified target's color changes exactly as requested across the video; shape, material, position, identity, background, lighting, and camera viewpoint are preserved.

Visual Quality & Stability
1. Severe artifacts, flicker, bleeding, or corrupted target.
2. Obvious color spill, unstable edges, or distracting temporal inconsistency.
3. Acceptable but with visible flicker, halos, or uneven recoloring.
4. Clean recoloring with only small artifacts.
5. Seamless, stable, physically plausible color edit throughout.

Consistency & Detail Fidelity
1. Unrelated objects, target shape/material, background, or camera viewpoint are heavily changed.
2. Important unrelated details or target geometry/material are changed.
3. Main scene is preserved but with noticeable unintended changes.
4. Nearly all unrelated details, target geometry, material, and motion are preserved.
5. Only the requested color changes; all other details are perfectly preserved.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


TEXTURE_CHANGE = """
You are a data rater specializing in strict local video texture and material edits. You will be given two videos (before and after editing) and the editing instruction.

Instruction Compliance
1. The requested material/texture change is absent, targets the wrong object, changes the whole scene, or makes an unrelated edit.
2. The texture/material is only weakly attempted, wrong, or appears on unrelated regions.
3. The requested texture/material is recognizable but incomplete, patchy, or mixed with unwanted color/shape/background changes.
4. The texture/material is correct with only minor scale, pattern, boundary, or temporal inaccuracies.
5. Only the specified target's texture/material changes exactly as requested across the video; shape, position, identity, background, lighting, and camera viewpoint are preserved.

Visual Quality & Stability
1. Severe artifacts, flicker, broken surface, or unusable video.
2. Obvious texture swimming, seams, resolution mismatch, or unstable boundaries.
3. Acceptable but with visible pattern instability or lighting mismatch.
4. Clean material/texture edit with only small artifacts.
5. Seamless, stable, physically plausible material/texture throughout.

Consistency & Detail Fidelity
1. Unrelated objects, target shape, position, background, or camera viewpoint are heavily changed.
2. Important unrelated details or target geometry are changed.
3. Main scene is preserved but with noticeable unintended changes.
4. Nearly all unrelated details, target geometry, position, and motion are preserved.
5. Only the requested texture/material changes; all other details are perfectly preserved.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


SHAPE_CHANGE = """
You are a data rater specializing in strict local video shape edits. You will be given two videos (before and after editing) and the editing instruction.

Instruction Compliance
1. The requested shape change is absent, targets the wrong object, replaces the object with unrelated content, changes the whole scene, or changes the camera.
2. The shape change is weak, wrong, or only appears in a few frames; unrelated regions are also changed.
3. The requested shape is recognizable but incomplete, unstable, or mixed with unwanted color/material/background changes.
4. The target shape is correct with only minor geometry, boundary, or temporal inaccuracies.
5. Only the specified target's shape/silhouette changes exactly as requested across the video; target identity/category, material/color where possible, position, background, lighting, and camera viewpoint are preserved.

Visual Quality & Stability
1. Severe warping, flicker, broken geometry, duplicated target, or unusable video.
2. Obvious unstable boundaries, holes, distorted perspective, or distracting temporal artifacts.
3. Acceptable but with visible geometry wobble, edge artifacts, or imperfect integration.
4. Clean shape edit with only small artifacts.
5. Seamless, stable, physically plausible shape edit throughout.

Consistency & Detail Fidelity
1. Unrelated objects, target position, background, or camera viewpoint are heavily changed.
2. Important unrelated details, target identity/category, material, or camera viewpoint are changed.
3. Main scene is preserved but with noticeable unintended changes.
4. Nearly all unrelated details, target position, material/color, and motion are preserved.
5. Only the requested shape changes; all other details are perfectly preserved.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
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


RELIGHT = """
You are a data rater specializing in grading video relighting edits. You will be given two videos (before and after editing) and the editing instruction.

Instruction Compliance
1. Lighting is unchanged, changed in the wrong direction/style, or the edit is unrelated.
2. The requested lighting appears only weakly, in the wrong area, or with major style mismatch.
3. The requested lighting is recognizable but incomplete, inconsistent, or mixed with incorrect illumination.
4. The requested lighting is applied well with only minor intensity, direction, or color inaccuracies.
5. The lighting exactly matches the requested style and direction while preserving the original scene content.

Visual Quality & Stability
1. Severe flicker, artifacts, blown-out regions, or unusable video.
2. Distracting temporal instability, inconsistent shadows, or obvious relighting artifacts.
3. Mostly stable but with visible lighting flicker, halos, or local artifacts.
4. Stable and clean relighting with only minor artifacts.
5. Perfectly stable, natural illumination, shadows, and color across the whole video.

Consistency & Detail Fidelity
1. Objects, identity, materials, motion, or camera viewpoint are heavily changed.
2. Important scene details or geometry are changed while relighting.
3. Main content is preserved but with noticeable unintended changes.
4. Nearly all unrelated details, motion, and camera viewpoint are preserved.
5. Only lighting changes; all scene content, motion, materials, and camera viewpoint are perfectly preserved.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


ACTION_EDIT = """
You are a data rater specializing in grading video edits that change a visible person's pose, gesture, facial expression, or gaze while preserving the camera.

Instruction Compliance
1. The requested action is absent, targets the wrong subject, changes the camera, or makes an unrelated edit.
2. The requested action is weakly attempted, changes the wrong body part/expression, or introduces camera movement.
3. The action is recognizable but inaccurate in direction, degree, limb, expression, or consistency.
4. The requested action is correct with only minor pose, anatomy, expression, or temporal inaccuracies.
5. The requested action is executed precisely and consistently while keeping the camera unchanged.

Visual Quality & Stability
1. Severe artifacts, broken anatomy, unusable face/hands, flicker, or major video corruption.
2. Obvious anatomical distortions, face damage, jitter, or distracting artifacts.
3. Acceptable but with visible local artifacts, imperfect anatomy, or temporal instability.
4. Clean and stable result with only minor artifacts.
5. Natural, clean, realistic, and temporally stable action edit.

Consistency & Detail Fidelity
1. Background, identity, clothing, camera viewpoint, or unrelated objects are heavily changed.
2. Important unrelated details are changed or the person identity/camera viewpoint is unstable.
3. Main content is preserved but with noticeable unintended changes.
4. Nearly all unrelated details and camera viewpoint are preserved.
5. Only the requested action changes; identity, clothing, background, and camera viewpoint are perfectly preserved.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


MOVE_EDIT = """
You are a data rater specializing in grading video edits that move a specified visible object or region within the same scene while preserving the camera.

Instruction Compliance
1. The specified target is not moved, moved in the wrong direction, the camera moves instead, or an unrelated edit is made.
2. The target moves only weakly, the wrong target moves, or camera viewpoint changes noticeably.
3. The target moves in the requested direction but with wrong magnitude, scale, placement, or temporal consistency.
4. The target is moved correctly with minor placement, fill, or temporal errors.
5. The specified target is moved precisely as instructed, with camera viewpoint unchanged.

Visual Quality & Stability
1. Severe artifacts, duplicated target, broken geometry, flicker, or unusable video.
2. Obvious paste marks, holes, shadows, jitter, or background fill artifacts.
3. Noticeable but tolerable fill, edge, shadow, or temporal issues.
4. Clean movement with only minor artifacts.
5. Seamless move with natural background fill, shadows, and temporal stability.

Consistency & Detail Fidelity
1. Unrelated objects, target identity, scale, background, or camera viewpoint are heavily changed.
2. Important unrelated details, target appearance, or camera viewpoint are changed.
3. Main scene is preserved but with visible unintended changes.
4. Nearly all unrelated details, target appearance, and camera viewpoint are preserved.
5. Only the target position changes; all unrelated details and camera viewpoint are perfectly preserved.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


SCALE_EDIT = """
You are a data rater specializing in grading video edits that make a specified visible object or region larger or smaller in place while preserving the camera.

Instruction Compliance
1. The target size is unchanged, changed in the wrong direction, the camera zooms instead, or an unrelated edit is made.
2. The size change is weakly attempted, targets the wrong object/region, or changes camera viewpoint.
3. The target changes size in the requested direction but with wrong magnitude, placement, shape, or temporal consistency.
4. The target size is changed correctly with minor scale, boundary, or temporal issues.
5. The specified target is resized exactly as instructed while staying in place and keeping the camera unchanged.

Visual Quality & Stability
1. Severe artifacts, broken target, duplicated parts, flicker, or unusable video.
2. Obvious paste marks, missing fill, distorted edges, shadows, jitter, or perspective errors.
3. Noticeable but tolerable fill, edge, perspective, or temporal issues.
4. Clean resize with only minor artifacts.
5. Seamless resize with natural background fill, shadows, perspective, and temporal stability.

Consistency & Detail Fidelity
1. Unrelated objects, target identity, background, or camera viewpoint are heavily changed.
2. Important unrelated details, target appearance, or camera viewpoint are changed.
3. Main scene is preserved but with visible unintended changes.
4. Nearly all unrelated details, target appearance, and camera viewpoint are preserved.
5. Only the target size changes; all unrelated details and camera viewpoint are perfectly preserved.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


DEFAULT_API_VERSION = "2024-03-01-preview"
DEFAULT_MODEL_ID = "gemini-2.5-pro"

# 只有提示词逐字来自官方 Kiwi-Edit 的类别才放进 prompt_type。
# 打分默认只走这个字典，所以不可能静默用上非官方的评分标准。
prompt_type = {
    "global_style": GLOBAL_STYLE,
    "local_change": LOCAL_CHANGE,
    "background_change": BACKGROUND_CHANGE,
    "local_remove": LOCAL_REMOVE,
    "local_add": LOCAL_ADD,
    # 以下键官方 prompt_type 里没有，但常量本身是官方的（官方脚本遇到这些
    # 类别会直接 KeyError）。OpenVE-Bench 的七类计划需要这两个。
    "creative_edit": CREATIVE_EDIT,
    "subtitle_edit": SUBTITLES_EDIT,
    "subtitles_edit": SUBTITLES_EDIT,
    "camera_edit": CAMERA_MULTI_SHOT_EDIT,
    "camera_multi_shot_edit": CAMERA_MULTI_SHOT_EDIT,
}

# 这些类别官方仓库里没有对应提示词，是 I2V-transfer 自行补的。
# 用它们打出来的分不能和官方口径的分放在一起比较，所以单独放，
# 需要时由调用方显式合并。
NON_OFFICIAL_PROMPT_TYPE = {
    "color_change": COLOR_CHANGE,
    "texture_change": TEXTURE_CHANGE,
    "shape_change": SHAPE_CHANGE,
    "relight": RELIGHT,
    "action_edit": ACTION_EDIT,
    "move_edit": MOVE_EDIT,
    "scale_edit": SCALE_EDIT,
}
