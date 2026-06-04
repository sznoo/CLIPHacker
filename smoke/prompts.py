# # prompts.py
BASE_PROMPT = {
    "prompt_id": "baseline",
    "prompt_name": "Baseline",
    "text": "Describe the image in one concise sentence.",
}
# PROMPTS = [
#     {
#         "prompt_id": "baseline",
#         "prompt_name": "Baseline",
#         "text": "Describe the image in one concise sentence.",
#     },
#     {
#         "prompt_id": "object_action",
#         "prompt_name": "Object-action",
#         "text": "Describe the image by naming the main visible subject, action, and setting in one sentence.",
#     },
#     {
#         "prompt_id": "grounded_detail",
#         "prompt_name": "Grounded detail",
#         "text": "Generate a visually grounded caption. Mention only objects, actions, and scene details clearly visible in the image.",
#     },
#     {
#         "prompt_id": "anti_generic",
#         "prompt_name": "Anti-generic",
#         "text": "Describe the image with concrete visible objects and actions. Avoid generic descriptions.",
#     },
#     {
#         "prompt_id": "main_subject_first",
#         "prompt_name": "Main subject first",
#         "text": "Describe the image in one sentence. Start with the main subject, then describe the action and surrounding scene.",
#     },
#     {
#         "prompt_id": "clip_friendly",
#         "prompt_name": "CLIP-friendly",
#         "text": "Write a concise caption that closely matches the visible content of the image, including the main subject, action, and scene.",
#     },
#     {
#         "prompt_id": "no_inference",
#         "prompt_name": "No inference",
#         "text": "Describe only what can be directly seen in the image. Do not infer unseen events, intentions, or identities.",
#     },
#     {
#         "prompt_id": "dense_visual",
#         "prompt_name": "Dense visual caption",
#         "text": "Describe the image in one specific sentence including the main subject, action, setting, and important visible objects.",
#     },
#     {
#         "prompt_id": "short_phrase",
#         "prompt_name": "Short phrase",
#         "text": "Describe the image in a short phrase focusing on the main visible subject and action.",
#     },
#     {
#         "prompt_id": "two_sentence_detail",
#         "prompt_name": "Two-sentence detail",
#         "text": "Describe the image in two short sentences. Mention the main subject, action, setting, and key visible objects.",
#     },

#     # New prompts from qualitative analysis
#     {
#         "prompt_id": "salient_anchor",
#         "prompt_name": "Salient anchor",
#         "text": "Describe the image by first identifying the most visually distinctive subject, object, or scene anchor, then add the key visible action and setting in one sentence.",
#     },
#     {
#         "prompt_id": "discriminative_detail",
#         "prompt_name": "Discriminative detail",
#         "text": "Write a caption that distinguishes this image from a generic image of the same scene by mentioning the most specific visible objects, colors, actions, or spatial details.",
#     },
#     {
#         "prompt_id": "anchor_then_context",
#         "prompt_name": "Anchor then context",
#         "text": "Describe the main visual anchor first, then briefly describe the supporting objects, people, and background context that make the scene recognizable.",
#     },
#     {
#         "prompt_id": "scene_state",
#         "prompt_name": "Scene state",
#         "text": "Describe the current state of the scene using concrete visible evidence, such as object placement, ongoing actions, clutter, traces of activity, or environmental details.",
#     },
#     {
#         "prompt_id": "role_structure",
#         "prompt_name": "Role structure",
#         "text": "For images with multiple people or objects, describe the central subject first, then explain the visible roles or interactions of the surrounding subjects in one concise caption.",
#     },
#     {
#         "prompt_id": "specific_not_verbose",
#         "prompt_name": "Specific not verbose",
#         "text": "Write one specific caption that includes the most important visible details, but avoid listing minor details that do not help identify the image.",
#     },
#     {
#         "prompt_id": "grounded_semantic",
#         "prompt_name": "Grounded semantic",
#         "text": "Describe the image with concrete visible details while also using reasonable scene-level terms when they are clearly supported by the image.",
#     },
#     {
#         "prompt_id": "avoid_mood_summary",
#         "prompt_name": "Avoid mood summary",
#         "text": "Avoid broad mood or event summaries. Instead, describe the visible subjects, objects, actions, and scene details that directly support the caption.",
#     },
#     {
#         "prompt_id": "uncertainty_aware",
#         "prompt_name": "Uncertainty-aware",
#         "text": "Describe the clearly visible subjects, objects, actions, and setting. Avoid confidently naming ambiguous object states or identities unless they are visually clear.",
#     },
#         {
#         "prompt_id": "ctx_main_anchor",
#         "prompt_name": "Context: main anchor",
#         "text": (
#             "Captioning context: The caption should be built around the most visually dominant anchor in the image, "
#             "such as the main person, object, action, or scene-defining structure. "
#             "Describe the image in one concise sentence using that anchor first."
#         ),
#     },
#     {
#         "prompt_id": "ctx_anchor_detail_context",
#         "prompt_name": "Context: anchor-detail-context",
#         "text": (
#             "Captioning context: Use a three-part structure: first the main visual anchor, "
#             "then the most concrete visible details, then only the background context needed to identify the scene. "
#             "Write one visually grounded caption."
#         ),
#     },
#     {
#         "prompt_id": "ctx_avoid_generic_scene",
#         "prompt_name": "Context: avoid generic scene",
#         "text": (
#             "Captioning context: Do not stop at a generic scene label such as a group of people, a meal, a store, or an event. "
#             "Include the concrete visible objects, actions, colors, or spatial details that distinguish this image. "
#             "Write one concise caption."
#         ),
#     },
#     {
#         "prompt_id": "ctx_foreground_priority",
#         "prompt_name": "Context: foreground priority",
#         "text": (
#             "Captioning context: If there is a clear foreground subject, describe that subject first, including what they are holding, wearing, or doing. "
#             "Then briefly mention the surrounding scene. Write one sentence."
#         ),
#     },
#     {
#         "prompt_id": "ctx_shared_scene_state",
#         "prompt_name": "Context: shared scene state",
#         "text": (
#             "Captioning context: For scenes with multiple people or many objects, focus on the shared visual state of the scene: "
#             "object placement, visible activity, food, tools, signs, equipment, clutter, or traces of action. "
#             "Write one grounded caption."
#         ),
#     },
#     {
#         "prompt_id": "ctx_supported_semantic",
#         "prompt_name": "Context: supported semantic",
#         "text": (
#             "Captioning context: Use scene-level terms when they are clearly supported by visible evidence, "
#             "such as historical costume, military-style display, rodeo arena, supermarket aisle, or dining table. "
#             "Avoid unsupported identities or intentions. Write one caption."
#         ),
#     },
#     {
#         "prompt_id": "ctx_uncertain_but_semantic",
#         "prompt_name": "Context: uncertain but semantic",
#         "text": (
#             "Captioning context: Describe concrete visible evidence and use reasonable scene-level grouping, "
#             "but avoid confidently naming ambiguous object states or hidden intentions. "
#             "Write one specific caption."
#         ),
#     },
#     {
#         "prompt_id": "ctx_clip_siglip_caption",
#         "prompt_name": "Context: CLIP/SigLIP caption",
#         "text": (
#             "Captioning context: Write a caption that would align well with image-text similarity models. "
#             "Prioritize visible nouns, actions, colors, object relations, and scene-defining details over mood or vague descriptions. "
#             "Use one fluent sentence."
#         ),
#     },
# ]

# PROMPTS = [
#     BASE_PROMPT,
#     # 1. main_subject_first 개선 계열
    
#     {
#         "prompt_id": "msf_concrete_context",
#         "prompt_name": "MSF concrete context",
#         "text": (
#             "Start with the main visible subject. In the same sentence, describe the subject's visible action, "
#             "one or two concrete attributes, and the surrounding scene context that helps identify the image."
#         ),
#     },
#     {
#         "prompt_id": "msf_object_relation",
#         "prompt_name": "MSF object relation",
#         "text": (
#             "Describe the image in one sentence by starting with the main subject, then clearly mentioning "
#             "the visible object they interact with and the scene where the interaction happens."
#         ),
#     },
#     {
#         "prompt_id": "msf_no_minor_details",
#         "prompt_name": "MSF no minor details",
#         "text": (
#             "Describe the main subject first, then the visible action and setting. Include only details that make "
#             "the image more identifiable, and avoid minor appearance details."
#         ),
#     },

#     # 2. ctx_uncertain_but_semantic 개선 계열
#     {
#         "prompt_id": "supported_scene_grouping",
#         "prompt_name": "Supported scene grouping",
#         "text": (
#             "Use concrete visible evidence to describe the image, but also use a clear scene-level phrase when "
#             "the objects, clothing, setting, or activity visibly support it. Avoid unsupported identities or intentions."
#         ),
#     },
#     {
#         "prompt_id": "careful_semantic_caption",
#         "prompt_name": "Careful semantic caption",
#         "text": (
#             "Write one specific caption using visible subjects, objects, actions, and setting. Use semantic terms "
#             "such as event, costume, market, meal, sport, or display only when they are clearly supported by the image."
#         ),
#     },
#     {
#         "prompt_id": "grounded_but_not_literal",
#         "prompt_name": "Grounded but not literal",
#         "text": (
#             "Describe what is visibly present, but do not reduce the image to a literal object list. "
#             "Group visible evidence into a natural scene description while avoiding guesses about hidden intent."
#         ),
#     },
#     {
#         "prompt_id": "semantic_with_uncertainty_control",
#         "prompt_name": "Semantic with uncertainty control",
#         "text": (
#             "Write a caption that combines visible details with supported scene understanding. "
#             "Do not confidently name ambiguous object states, identities, or emotions unless they are visually clear."
#         ),
#     },

#     # 3. CLIP/SigLIP 친화 계열
#     {
#         "prompt_id": "visual_keyword_sentence",
#         "prompt_name": "Visual keyword sentence",
#         "text": (
#             "Write one fluent caption containing the most important visible nouns, actions, colors, and scene words. "
#             "Avoid vague mood descriptions and avoid unnecessary minor details."
#         ),
#     },
#     {
#         "prompt_id": "siglip_visible_alignment",
#         "prompt_name": "SigLIP visible alignment",
#         "text": (
#             "Write a visually aligned caption. Prioritize concrete visible subjects, objects, actions, colors, "
#             "spatial relations, and setting over abstract descriptions."
#         ),
#     },
#     {
#         "prompt_id": "image_specific_caption",
#         "prompt_name": "Image-specific caption",
#         "text": (
#             "Write a caption that would distinguish this exact image from other images of the same general scene. "
#             "Mention the main subject, the key visible object or action, and one scene-specific detail."
#         ),
#     },

#     # 4. foreground/context routing 가능성 확인
#     {
#         "prompt_id": "foreground_then_background",
#         "prompt_name": "Foreground then background",
#         "text": (
#             "If the image has a clear foreground subject, describe it first with what it is doing or holding. "
#             "Then add the background scene only as supporting context."
#         ),
#     },
#     {
#         "prompt_id": "dominant_subject_or_scene",
#         "prompt_name": "Dominant subject or scene",
#         "text": (
#             "Decide whether the image is mainly about a dominant foreground subject or a broader scene. "
#             "Write one caption focused on whichever is more visually important."
#         ),
#     },
#     {
#         "prompt_id": "people_scene_balance",
#         "prompt_name": "People-scene balance",
#         "text": (
#             "For images with people, describe the visible people only as much as needed, then include the objects, "
#             "setting, and activity that define the scene."
#         ),
#     },

#     # 5. generic social/event summary 방지
#     {
#         "prompt_id": "avoid_generic_social",
#         "prompt_name": "Avoid generic social",
#         "text": (
#             "Do not summarize the image only as people gathering, eating, shopping, or enjoying an event. "
#             "Instead, name the visible objects, actions, and setting details that make the scene specific."
#         ),
#     },
#     {
#         "prompt_id": "specific_event_evidence",
#         "prompt_name": "Specific event evidence",
#         "text": (
#             "If the image shows an event or activity, describe the visible evidence for that event: clothing, objects, "
#             "tools, signs, food, equipment, body actions, or setting."
#         ),
#     },

#     # 6. detail 과잉 방지
#     {
#         "prompt_id": "salient_details_only",
#         "prompt_name": "Salient details only",
#         "text": (
#             "Write one grounded caption using only salient visible details: the main subject, key action, important objects, "
#             "and setting. Do not list small details unless they help identify the image."
#         ),
#     },
#     {
#         "prompt_id": "compact_grounded_detail",
#         "prompt_name": "Compact grounded detail",
#         "text": (
#             "Write a compact but specific caption. Include the main subject, visible action, setting, and two or three "
#             "image-defining details."
#         ),
#     },

#     # 7. 예시 포함 프롬프트: 실험 반경 확장용
#     {
#         "prompt_id": "examples_anchor_caption",
#         "prompt_name": "Examples anchor caption",
#         "text": (
#             "Write one visually grounded caption. Focus on the image-defining anchor, not a generic scene label. "
#             "For example, prefer 'a man in a suit holding a blue basket in a supermarket aisle' over "
#             "'a person shopping', and prefer 'a cluttered table with plates and a salad bowl' over 'people eating together'."
#         ),
#     },
#     {
#         "prompt_id": "examples_semantic_caution",
#         "prompt_name": "Examples semantic caution",
#         "text": (
#             "Describe concrete visible evidence and use supported scene terms. For example, visible uniforms and equipment "
#             "can support 'military-style display', but do not infer identities, emotions, or ambiguous object states. "
#             "Write one specific caption."
#         ),
#     },

#     # 8. deliberately broader context prompt
#     {
#         "prompt_id": "caption_for_retrieval",
#         "prompt_name": "Caption for retrieval",
#         "text": (
#             "Write a caption useful for retrieving this image from a large image database. Include the main visible subject, "
#             "distinctive objects, action, setting, and any clearly visible colors or spatial relations."
#         ),
#     },
# ]
# PROMPTS = [
#     # Core baselines
#     {
#         "prompt_id": "baseline",
#         "prompt_name": "Baseline",
#         "text": "Describe the image in one concise sentence.",
#     },
#     {
#         "prompt_id": "main_subject_first",
#         "prompt_name": "Main subject first",
#         "text": "Describe the image in one sentence. Start with the main subject, then describe the action and surrounding scene.",
#     },
#     {
#         "prompt_id": "clip_friendly",
#         "prompt_name": "CLIP-friendly",
#         "text": "Write a concise caption that closely matches the visible content of the image, including the main subject, action, and scene.",
#     },
#     {
#         "prompt_id": "dense_visual",
#         "prompt_name": "Dense visual caption",
#         "text": "Describe the image in one specific sentence including the main subject, action, setting, and important visible objects.",
#     },
#     {
#         "prompt_id": "no_inference",
#         "prompt_name": "No inference",
#         "text": "Describe only what can be directly seen in the image. Do not infer unseen events, intentions, or identities.",
#     },

#     # Current strong prompts
#     {
#         "prompt_id": "semantic_with_uncertainty_control",
#         "prompt_name": "Semantic with uncertainty control",
#         "text": (
#             "Write a caption that combines visible details with supported scene understanding. "
#             "Do not confidently name ambiguous object states, identities, or emotions unless they are visually clear."
#         ),
#     },
#     {
#         "prompt_id": "careful_semantic_caption",
#         "prompt_name": "Careful semantic caption",
#         "text": (
#             "Write one specific caption using visible subjects, objects, actions, and setting. "
#             "Use semantic terms such as event, costume, market, meal, sport, or display only when they are clearly supported by the image."
#         ),
#     },
#     {
#         "prompt_id": "people_scene_balance",
#         "prompt_name": "People-scene balance",
#         "text": (
#             "For images with people, describe the visible people only as much as needed, then include the objects, "
#             "setting, and activity that define the scene."
#         ),
#     },
#     {
#         "prompt_id": "caption_for_retrieval",
#         "prompt_name": "Caption for retrieval",
#         "text": (
#             "Write a caption useful for retrieving this image from a large image database. Include the main visible subject, "
#             "distinctive objects, action, setting, and any clearly visible colors or spatial relations."
#         ),
#     },
#     {
#         "prompt_id": "compact_grounded_detail",
#         "prompt_name": "Compact grounded detail",
#         "text": (
#             "Write a compact but specific caption. Include the main subject, visible action, setting, and two or three "
#             "image-defining details."
#         ),
#     },

#     # Branch-style prompts
#     {
#         "prompt_id": "branch_foreground_vs_scene",
#         "prompt_name": "Branch: foreground vs scene",
#         "text": (
#             "Describe the image in one sentence. "
#             "If the image has a clear dominant foreground subject, describe that subject first, including what it is doing, holding, or wearing, "
#             "then add the surrounding scene as brief context. "
#             "If the image does not have a single dominant foreground subject, describe the overall scene using the most important visible objects, actions, and setting details. "
#             "Avoid generic summaries."
#         ),
#     },
#     {
#         "prompt_id": "branch_single_vs_multi_people",
#         "prompt_name": "Branch: single vs multi people",
#         "text": (
#             "Write one visually grounded caption. "
#             "If one person is clearly the main subject, focus on that person first, then mention the visible scene context. "
#             "If multiple people are similarly important, describe the shared scene, visible interactions, and the objects or setting that define the activity. "
#             "Use concrete visible details and avoid vague social summaries."
#         ),
#     },
#     {
#         "prompt_id": "branch_semantic_uncertainty",
#         "prompt_name": "Branch: semantic and uncertainty",
#         "text": (
#             "Write one specific caption using visible subjects, objects, actions, and setting. "
#             "If the scene clearly supports a scene-level description such as a meal, a market, a performance, a sports scene, or a display, use that supported scene description. "
#             "If some object states or identities are ambiguous, describe them cautiously using only what is clearly visible. "
#             "Avoid unsupported intentions, emotions, or identities."
#         ),
#     },
#     {
#         "prompt_id": "branch_distinctive_object",
#         "prompt_name": "Branch: distinctive object",
#         "text": (
#             "Describe the image in one sentence. "
#             "If there is a visually distinctive object or structure that strongly defines the image, mention it explicitly along with the main subject or action. "
#             "Otherwise, focus on the main visible subject, action, and setting. "
#             "Prefer concrete visible nouns, actions, and scene details over vague mood descriptions."
#         ),
#     },
#     {
#         "prompt_id": "branch_clean_vs_cluttered_scene",
#         "prompt_name": "Branch: clean vs cluttered scene",
#         "text": (
#             "Write one grounded caption. "
#             "If the image contains many visible objects or a cluttered shared scene, include the key objects and the visible state of the scene that make it distinctive. "
#             "If the image is visually simple, focus on the main subject, action, and setting. "
#             "Do not include minor details unless they help identify the image."
#         ),
#     },
#     {
#         "prompt_id": "branch_retrieval_style",
#         "prompt_name": "Branch: retrieval style",
#         "text": (
#             "Write a caption that would help retrieve this exact image. "
#             "If the image is defined mainly by a person or object in the foreground, mention that subject first and add the key surrounding context. "
#             "If the image is defined mainly by a broader scene or activity, describe the most important visible objects, actions, and setting details. "
#             "Use concrete visible evidence and avoid generic summaries."
#         ),
#     },
#     {
#         "prompt_id": "branch_supported_semantic",
#         "prompt_name": "Branch: supported semantic",
#         "text": (
#             "Describe the image with visible evidence first. "
#             "If the visible clothing, objects, setting, or activity clearly indicate a broader semantic scene, include that scene description. "
#             "If the broader scene is unclear, stay with concrete visible subjects, objects, actions, and setting."
#         ),
#     },
#     {
#         "prompt_id": "branch_with_examples",
#         "prompt_name": "Branch: with examples",
#         "text": (
#             "Write one visually grounded caption. "
#             "If one subject dominates the image, describe that subject first and then the scene context. "
#             "If the image is a shared scene, describe the scene-defining objects and activity rather than a generic summary. "
#             "For example, prefer 'a man in a suit holding a blue basket in a supermarket aisle' over 'a person shopping', "
#             "and prefer 'a cluttered dining table with plates and a salad bowl' over 'people eating together'."
#         ),
#     },
# ]

# PROMPTS = [
#         BASE_PROMPT,
#     {
#         "prompt_id": "sem_uncertainty_main_subject",
#         "prompt_name": "Semantic uncertainty main subject",
#         "text": (
#             "Write one specific caption. Start with the main visible subject or scene, "
#             "include concrete visible details, and use supported scene-level terms only when the image clearly supports them. "
#             "Avoid ambiguous identities, emotions, or object states."
#         ),
#     },
#     {
#         "prompt_id": "supported_scene_one_sentence",
#         "prompt_name": "Supported scene one sentence",
#         "text": (
#             "Describe the image in one sentence using visible evidence. "
#             "Mention the main subject, visible action, setting, and any clearly supported scene category such as a meal, market, sport, display, or event."
#         ),
#     },
#     {
#         "prompt_id": "main_subject_supported_context",
#         "prompt_name": "Main subject supported context",
#         "text": (
#             "Start with the main subject, then describe the visible action and the surrounding context. "
#             "Use only details that are clearly visible or strongly supported by visible objects, clothing, setting, or activity."
#         ),
#     },
#     {
#         "prompt_id": "safe_semantic_retrieval",
#         "prompt_name": "Safe semantic retrieval",
#         "text": (
#             "Write a caption useful for matching this image in a retrieval system. "
#             "Include the main visible subject, action, setting, and distinctive visible objects, while avoiding unsupported guesses."
#         ),
#     },
#     {
#         "prompt_id": "distinctive_subject_object",
#         "prompt_name": "Distinctive subject-object",
#         "text": (
#             "Describe the image by focusing on the most distinctive visible subject or object, "
#             "then add the action, setting, and one or two concrete details that make the image specific."
#         ),
#     },
#     {
#         "prompt_id": "foreground_or_shared_scene",
#         "prompt_name": "Foreground or shared scene",
#         "text": (
#             "If one foreground subject dominates the image, describe that subject first with its visible action or object. "
#             "Otherwise, describe the shared scene using the visible objects, actions, and setting that define it. "
#             "Write one sentence."
#         ),
#     },
#     {
#         "prompt_id": "visible_evidence_caption",
#         "prompt_name": "Visible evidence caption",
#         "text": (
#             "Write one caption grounded in visible evidence: main subject, visible action, important objects, setting, and supported scene meaning. "
#             "Avoid vague mood words and avoid unsupported identities."
#         ),
#     },
#     {
#         "prompt_id": "specific_scene_without_overdetail",
#         "prompt_name": "Specific scene without overdetail",
#         "text": (
#             "Write a specific but compact caption. "
#             "Mention the main subject or scene, the visible action, and the most important identifying details, but do not list minor background details."
#         ),
#     },
#     {
#         "prompt_id": "object_relation_scene",
#         "prompt_name": "Object relation scene",
#         "text": (
#             "Describe the image in one sentence by naming the main subject, the visible object or person it interacts with, "
#             "and the scene where the interaction occurs."
#         ),
#     },
#     {
#         "prompt_id": "scene_defining_details",
#         "prompt_name": "Scene-defining details",
#         "text": (
#             "Write a caption that captures the scene-defining details: the main subject, key visible objects, action, setting, and any supported activity or event type."
#         ),
#     },
#     {
#         "prompt_id": "avoid_generic_with_support",
#         "prompt_name": "Avoid generic with support",
#         "text": (
#             "Avoid generic summaries such as people at an event, people eating, or a person standing. "
#             "Instead, describe the visible evidence that supports the scene: subjects, objects, actions, clothing, setting, or equipment."
#         ),
#     },
#     {
#         "prompt_id": "balanced_visual_semantic",
#         "prompt_name": "Balanced visual semantic",
#         "text": (
#             "Write one balanced caption that combines concrete visible details with a natural scene description. "
#             "Be specific enough to identify the image, but do not infer hidden intentions, emotions, or exact identities."
#         ),
#     },
#     {
#         "prompt_id": "central_subject_or_scene_anchor",
#         "prompt_name": "Central subject or scene anchor",
#         "text": (
#             "Identify whether the image is centered on a main subject or a broader scene, then write one caption around that visual center. "
#             "Include visible objects, actions, and setting details that support it."
#         ),
#     },
#     {
#         "prompt_id": "retrieval_compact",
#         "prompt_name": "Retrieval compact",
#         "text": (
#             "Write a compact image-retrieval caption with concrete visible nouns, actions, colors, and setting words. "
#             "Keep it fluent and avoid long lists or vague mood descriptions."
#         ),
#     },
#     {
#         "prompt_id": "supported_activity_caption",
#         "prompt_name": "Supported activity caption",
#         "text": (
#             "If the image shows an activity, describe the visible evidence for that activity, including subjects, objects, body actions, setting, and equipment. "
#             "If no activity is clear, describe the main visible subject and scene."
#         ),
#     },
#     {
#         "prompt_id": "uncertainty_controlled_detail",
#         "prompt_name": "Uncertainty controlled detail",
#         "text": (
#             "Write one detailed caption using clearly visible subjects, objects, actions, and setting. "
#             "Use cautious wording for ambiguous objects or scene meanings, and avoid unsupported emotions or identities."
#         ),
#     },
#     {
#         "prompt_id": "main_action_scene_specific",
#         "prompt_name": "Main action scene specific",
#         "text": (
#             "Describe the main visible action first, then the subject performing it and the scene-specific objects or setting around it. "
#             "Use one concise sentence."
#         ),
#     },
#     {
#         "prompt_id": "caption_like_human_observer",
#         "prompt_name": "Caption like human observer",
#         "text": (
#             "Write one caption as a careful human observer would: state what is visibly happening, name the important subjects and objects, "
#             "and include only scene meanings that are clearly supported."
#         ),
#     },
# ]
# PROMPTS = [
#     # 1. caption_like_human_observer 직접 변형
#     BASE_PROMPT,
#     {
#         "prompt_id": "observer_visible_happening",
#         "prompt_name": "Observer visible happening",
#         "text": (
#             "Write one caption as a careful observer would. "
#             "State what is visibly happening, name the important subjects and objects, "
#             "and keep the caption grounded in clear visual evidence."
#         ),
#     },
#     {
#         "prompt_id": "observer_supported_scene",
#         "prompt_name": "Observer supported scene",
#         "text": (
#             "Write one caption as a careful observer would. "
#             "Describe the visible subjects, objects, and action, and include a scene-level meaning only when it is clearly supported by the image."
#         ),
#     },
#     {
#         "prompt_id": "observer_no_unsupported_inference",
#         "prompt_name": "Observer no unsupported inference",
#         "text": (
#             "Write one caption as a careful human observer would. "
#             "Describe the important visible subjects, objects, actions, and setting, but avoid unsupported identities, emotions, intentions, or ambiguous object states."
#         ),
#     },
#     {
#         "prompt_id": "observer_compact_specific",
#         "prompt_name": "Observer compact specific",
#         "text": (
#             "Write one compact caption as a careful observer would. "
#             "Mention the main visible subject or scene, the action, and the most important objects or setting details."
#         ),
#     },

#     # 2. visible happening 계열
#     {
#         "prompt_id": "visible_happening_subject_object",
#         "prompt_name": "Visible happening subject-object",
#         "text": (
#             "Describe what is visibly happening in one sentence. "
#             "Include the main subject, the object or person involved, the visible action, and the setting."
#         ),
#     },
#     {
#         "prompt_id": "visible_happening_scene_context",
#         "prompt_name": "Visible happening scene context",
#         "text": (
#             "Write one caption that explains what is visibly happening in the image. "
#             "Use concrete subjects, objects, actions, and setting details, with only enough context to identify the scene."
#         ),
#     },
#     {
#         "prompt_id": "visible_action_supported_context",
#         "prompt_name": "Visible action supported context",
#         "text": (
#             "Start from the visible action or activity in the image, then describe the main subject, important objects, and supported scene context in one sentence."
#         ),
#     },

#     # 3. important subjects/objects 계열
#     {
#         "prompt_id": "important_subjects_objects",
#         "prompt_name": "Important subjects and objects",
#         "text": (
#             "Write one caption focused on the important visible subjects and objects in the image. "
#             "Describe their visible action or relation and the setting that makes the scene clear."
#         ),
#     },
#     {
#         "prompt_id": "important_not_minor",
#         "prompt_name": "Important not minor",
#         "text": (
#             "Describe the image using the important visible subjects, objects, action, and setting. "
#             "Do not list minor background details unless they help identify the image."
#         ),
#     },
#     {
#         "prompt_id": "subject_object_scene_meaning",
#         "prompt_name": "Subject object scene meaning",
#         "text": (
#             "Write one caption that names the main subject, key visible objects, and setting, "
#             "then expresses the supported scene meaning without adding unsupported details."
#         ),
#     },

#     # 4. supported semantic 계열
#     {
#         "prompt_id": "supported_semantic_compact",
#         "prompt_name": "Supported semantic compact",
#         "text": (
#             "Write one compact caption with concrete visible details and a supported scene description. "
#             "Avoid vague mood descriptions and unsupported guesses."
#         ),
#     },
#     {
#         "prompt_id": "supported_semantic_subject_first",
#         "prompt_name": "Supported semantic subject first",
#         "text": (
#             "Start with the main visible subject or scene. "
#             "Then include visible objects, actions, and a scene-level description only if the image clearly supports it."
#         ),
#     },
#     {
#         "prompt_id": "supported_semantic_no_mood",
#         "prompt_name": "Supported semantic no mood",
#         "text": (
#             "Describe the image with visible subjects, objects, actions, and setting. "
#             "Use supported scene-level terms, but avoid vague mood words such as lively, happy, casual, or enjoyable unless clearly visible."
#         ),
#     },

#     # 5. 평균 gain 안정화 목적
#     {
#         "prompt_id": "specific_but_cautious",
#         "prompt_name": "Specific but cautious",
#         "text": (
#             "Write one specific caption grounded in what is visible. "
#             "Include the main subject, action, setting, and important objects, while being cautious about ambiguous identities, emotions, or object states."
#         ),
#     },
#     {
#         "prompt_id": "specific_scene_careful_observer",
#         "prompt_name": "Specific scene careful observer",
#         "text": (
#             "As a careful observer, write one specific caption that identifies the visible scene through its main subject, action, important objects, and setting."
#         ),
#     },
#     {
#         "prompt_id": "concrete_scene_evidence",
#         "prompt_name": "Concrete scene evidence",
#         "text": (
#             "Write one caption using concrete scene evidence: visible subjects, objects, actions, clothing, equipment, signs, food, or setting details that identify the image."
#         ),
#     },

#     # 6. weak branch-lite, 너무 복잡하지 않게
#     {
#         "prompt_id": "dominant_or_shared_observer",
#         "prompt_name": "Dominant or shared observer",
#         "text": (
#             "Write one caption as a careful observer. "
#             "If one subject dominates, describe that subject and its action first; otherwise describe the shared scene through its key visible objects, actions, and setting."
#         ),
#     },
#     {
#         "prompt_id": "foreground_or_scene_evidence",
#         "prompt_name": "Foreground or scene evidence",
#         "text": (
#             "If there is a clear foreground subject, describe it first with its visible action or object. "
#             "If the image is defined by a broader scene, describe the visible evidence that defines that scene. "
#             "Write one sentence."
#         ),
#     },

#     # 7. 현재 best와 retrieval 방향의 절충
#     {
#         "prompt_id": "observer_retrieval_safe",
#         "prompt_name": "Observer retrieval safe",
#         "text": (
#             "Write one caption as a careful observer, useful for matching this image. "
#             "Include the main visible subject, action, setting, and distinctive objects, but avoid unsupported guesses or long detail lists."
#         ),
#     },
# ]
# PROMPTS = [
#     # Required baseline
#     {
#         "prompt_id": "baseline",
#         "prompt_name": "Baseline",
#         "text": "Describe the image in one concise sentence.",
#     },

#     # Current controls / previous strong prompts
#     {
#         "prompt_id": "caption_like_human_observer",
#         "prompt_name": "Caption like human observer",
#         "text": (
#             "Write one caption as a careful human observer would: state what is visibly happening, "
#             "name the important subjects and objects, and include only scene meanings that are clearly supported."
#         ),
#     },
#     {
#         "prompt_id": "semantic_with_uncertainty_control",
#         "prompt_name": "Semantic with uncertainty control",
#         "text": (
#             "Write a caption that combines visible details with supported scene understanding. "
#             "Do not confidently name ambiguous object states, identities, or emotions unless they are visually clear."
#         ),
#     },
#     {
#         "prompt_id": "main_subject_first",
#         "prompt_name": "Main subject first",
#         "text": "Describe the image in one sentence. Start with the main subject, then describe the action and surrounding scene.",
#     },

#     # Best-prompt variants
#     {
#         "prompt_id": "observer_compact_evidence",
#         "prompt_name": "Observer compact evidence",
#         "text": (
#             "Write one compact caption as a careful human observer would. "
#             "State what is visibly happening, name the important subjects and objects, "
#             "and keep any scene interpretation clearly supported by visual evidence."
#         ),
#     },
#     {
#         "prompt_id": "observer_subject_action_objects",
#         "prompt_name": "Observer subject-action-objects",
#         "text": (
#             "Write one caption as a careful observer would: describe the main visible subject, "
#             "the action taking place, the important objects, and the supported scene context."
#         ),
#     },
#     {
#         "prompt_id": "observer_identifying_details",
#         "prompt_name": "Observer identifying details",
#         "text": (
#             "Write one caption as a careful human observer would. "
#             "Include the visible subjects, actions, and the few identifying objects or details that make this image specific."
#         ),
#     },
#     {
#         "prompt_id": "observer_supported_scene_meaning",
#         "prompt_name": "Observer supported scene meaning",
#         "text": (
#             "Write one caption as a careful observer would. "
#             "Describe the visible subjects, objects, and actions, and include a broader scene meaning only when it is clearly supported."
#         ),
#     },
#     {
#         "prompt_id": "observer_no_guessing",
#         "prompt_name": "Observer no guessing",
#         "text": (
#             "Write one caption as a careful human observer would. "
#             "Describe what is visibly happening and the important visible objects, "
#             "without guessing identities, emotions, intentions, or ambiguous object states."
#         ),
#     },
#     {
#         "prompt_id": "observer_scene_specific",
#         "prompt_name": "Observer scene specific",
#         "text": (
#             "Write one scene-specific caption as a careful observer would. "
#             "Avoid generic summaries by naming the important visible subjects, objects, actions, and supported setting."
#         ),
#     },

#     # Synthesis prompts from previous successful patterns
#     {
#         "prompt_id": "visible_evidence_then_meaning",
#         "prompt_name": "Visible evidence then meaning",
#         "text": (
#             "Write one caption that first reflects the concrete visible evidence, then expresses the supported scene meaning. "
#             "Include the main subject, action, important objects, and setting."
#         ),
#     },
#     {
#         "prompt_id": "important_visual_elements",
#         "prompt_name": "Important visual elements",
#         "text": (
#             "Describe the image in one sentence using the important visual elements: "
#             "main subject, visible action, key objects, setting, and any clearly supported activity or scene type."
#         ),
#     },
#     {
#         "prompt_id": "specific_supported_caption",
#         "prompt_name": "Specific supported caption",
#         "text": (
#             "Write one specific caption grounded in visible subjects, objects, actions, and setting. "
#             "Use scene-level descriptions only when the image provides clear visual support."
#         ),
#     },
#     {
#         "prompt_id": "human_observer_retrieval_light",
#         "prompt_name": "Human observer retrieval light",
#         "text": (
#             "Write one caption as a careful observer, useful for matching this image. "
#             "Mention the main visible subject, action, setting, and one or two distinctive visible details."
#         ),
#     },
#     {
#         "prompt_id": "subject_or_scene_center",
#         "prompt_name": "Subject or scene center",
#         "text": (
#             "Write one caption around the visual center of the image. "
#             "If a subject dominates, describe the subject and action; if the broader scene dominates, describe the key objects, actions, and setting."
#         ),
#     },
#     {
#         "prompt_id": "supported_activity_observer",
#         "prompt_name": "Supported activity observer",
#         "text": (
#             "As a careful observer, describe the visible activity in the image. "
#             "Name the subjects, objects, body actions, equipment, or setting details that support that activity."
#         ),
#     },
#     {
#         "prompt_id": "not_generic_not_overdetailed",
#         "prompt_name": "Not generic not overdetailed",
#         "text": (
#             "Write one caption that is specific but not overly detailed. "
#             "Avoid generic scene summaries, but include only the visible details that help identify the image."
#         ),
#     },
#     {
#         "prompt_id": "clear_visual_support",
#         "prompt_name": "Clear visual support",
#         "text": (
#             "Write one caption using only clearly supported visual information. "
#             "Mention the main subject or scene, visible action, important objects, and setting in a natural sentence."
#         ),
#     },

#     # Slightly stronger variants, useful to test whether richer instruction helps
#     {
#         "prompt_id": "observer_three_part",
#         "prompt_name": "Observer three-part",
#         "text": (
#             "Write one caption as a careful observer. "
#             "Use three elements: what is happening, who or what is important, and which visible objects or setting details define the scene."
#         ),
#     },
#     {
#         "prompt_id": "observer_distinctive_but_safe",
#         "prompt_name": "Observer distinctive but safe",
#         "text": (
#             "Write one caption as a careful observer. "
#             "Choose the most distinctive visible subjects, objects, or actions, but avoid unsupported guesses about identity, emotion, or intention."
#         ),
#     },
#     {
#         "prompt_id": "observer_grounded_scene_caption",
#         "prompt_name": "Observer grounded scene caption",
#         "text": (
#             "Write one grounded scene caption. "
#             "A careful observer should be able to verify the subjects, objects, actions, and scene meaning directly from the image."
#         ),
#     },
# ]
PROMPTS = [
    {
        "prompt_id": "baseline",
        "prompt_name": "Baseline",
        "text": "Describe the image in one concise sentence.",
    },
    {
        "prompt_id": "caption_like_human_observer",
        "prompt_name": "Caption like human observer",
        "text": (
            "Write one caption as a careful human observer would: state what is visibly happening, "
            "name the important subjects and objects, and include only scene meanings that are clearly supported."
        ),
    },
    {
        "prompt_id": "semantic_with_uncertainty_control",
        "prompt_name": "Semantic with uncertainty control",
        "text": (
            "Write a caption that combines visible details with supported scene understanding. "
            "Do not confidently name ambiguous object states, identities, or emotions unless they are visually clear."
        ),
    },
    {
        "prompt_id": "main_subject_first",
        "prompt_name": "Main subject first",
        "text": "Describe the image in one sentence. Start with the main subject, then describe the action and surrounding scene.",
    },

    # Minimal local edits around caption_like_human_observer
    {
        "prompt_id": "observer_visible_evidence_supported",
        "prompt_name": "Observer visible evidence supported",
        "text": (
            "Write one caption as a careful human observer would: state what is visibly happening, "
            "name the important subjects and objects, and include only scene meanings clearly supported by visible evidence."
        ),
    },
    {
        "prompt_id": "observer_important_visible_elements",
        "prompt_name": "Observer important visible elements",
        "text": (
            "Write one caption as a careful human observer would: state what is visibly happening, "
            "name the important visible subjects and objects, and include only scene meanings that are clearly supported."
        ),
    },
    {
        "prompt_id": "observer_natural_supported",
        "prompt_name": "Observer natural supported",
        "text": (
            "Write one natural caption as a careful human observer would: state what is visibly happening, "
            "name the important subjects and objects, and include only scene meanings that are clearly supported."
        ),
    },
    {
        "prompt_id": "observer_specific_supported",
        "prompt_name": "Observer specific supported",
        "text": (
            "Write one specific caption as a careful human observer would: state what is visibly happening, "
            "name the important subjects and objects, and include only scene meanings that are clearly supported."
        ),
    },
    {
        "prompt_id": "observer_grounded_supported",
        "prompt_name": "Observer grounded supported",
        "text": (
            "Write one visually grounded caption as a careful human observer would: state what is visibly happening, "
            "name the important subjects and objects, and include only scene meanings that are clearly supported."
        ),
    },

    # Slightly stronger object/subject emphasis
    {
        "prompt_id": "observer_subject_object_action",
        "prompt_name": "Observer subject object action",
        "text": (
            "Write one caption as a careful human observer would: state what is visibly happening, "
            "name the important subjects, objects, and actions, and include only scene meanings that are clearly supported."
        ),
    },
    {
        "prompt_id": "observer_key_objects_supported",
        "prompt_name": "Observer key objects supported",
        "text": (
            "Write one caption as a careful human observer would: state what is visibly happening, "
            "name the important subjects and key visible objects, and include only scene meanings that are clearly supported."
        ),
    },
    {
        "prompt_id": "observer_subjects_objects_setting",
        "prompt_name": "Observer subjects objects setting",
        "text": (
            "Write one caption as a careful human observer would: state what is visibly happening, "
            "name the important subjects and objects, mention the visible setting, and include only scene meanings that are clearly supported."
        ),
    },

    # Anti-generic but not too aggressive
    {
        "prompt_id": "observer_not_generic",
        "prompt_name": "Observer not generic",
        "text": (
            "Write one caption as a careful human observer would: avoid generic summaries, state what is visibly happening, "
            "name the important subjects and objects, and include only scene meanings that are clearly supported."
        ),
    },
    {
        "prompt_id": "observer_identifiable_scene",
        "prompt_name": "Observer identifiable scene",
        "text": (
            "Write one caption as a careful human observer would: state what is visibly happening, "
            "name the important subjects and objects that make the image identifiable, and include only clearly supported scene meanings."
        ),
    },
    {
        "prompt_id": "observer_specific_not_list",
        "prompt_name": "Observer specific not list",
        "text": (
            "Write one caption as a careful human observer would: state what is visibly happening, "
            "name the important subjects and objects, and include only clearly supported scene meanings without making a long list of details."
        ),
    },

    # Supported semantic + observer hybrid
    {
        "prompt_id": "observer_supported_scene_terms",
        "prompt_name": "Observer supported scene terms",
        "text": (
            "Write one caption as a careful human observer would: state what is visibly happening, "
            "name the important subjects and objects, and use scene-level terms only when clearly supported by the visible image."
        ),
    },
    {
        "prompt_id": "observer_cautious_scene_meaning",
        "prompt_name": "Observer cautious scene meaning",
        "text": (
            "Write one caption as a careful human observer would: state what is visibly happening, "
            "name the important subjects and objects, and include broader scene meaning only when the image clearly supports it."
        ),
    },
    {
        "prompt_id": "observer_no_hidden_inference",
        "prompt_name": "Observer no hidden inference",
        "text": (
            "Write one caption as a careful human observer would: state what is visibly happening, "
            "name the important subjects and objects, and avoid hidden guesses about identities, intentions, emotions, or ambiguous object states."
        ),
    },

    # Slight branch-lite variants but still close to best prompt
    {
        "prompt_id": "observer_subject_or_scene",
        "prompt_name": "Observer subject or scene",
        "text": (
            "Write one caption as a careful human observer would: if a subject dominates, state what that subject is doing; "
            "if the broader scene dominates, state what is visibly happening in the scene. "
            "Name important subjects and objects and include only clearly supported scene meanings."
        ),
    },
    {
        "prompt_id": "observer_people_or_objects",
        "prompt_name": "Observer people or objects",
        "text": (
            "Write one caption as a careful human observer would: state what is visibly happening, "
            "focusing on the important people if people dominate the image or the important objects if objects define the scene, "
            "and include only clearly supported scene meanings."
        ),
    },

    # Very close final candidates
    {
        "prompt_id": "observer_best_plus_setting",
        "prompt_name": "Observer best plus setting",
        "text": (
            "Write one caption as a careful human observer would: state what is visibly happening, "
            "name the important subjects and objects, mention the setting when visible, "
            "and include only scene meanings that are clearly supported."
        ),
    },
    {
        "prompt_id": "observer_best_plus_specific",
        "prompt_name": "Observer best plus specific",
        "text": (
            "Write one specific caption as a careful human observer would: state what is visibly happening, "
            "name the important subjects and objects, mention the setting, "
            "and include only scene meanings clearly supported by visible evidence."
        ),
    },
]
def get_prompts():
    return PROMPTS


def get_prompt_ids():
    return [p["prompt_id"] for p in PROMPTS]


def get_prompt_text(prompt_id: str) -> str:
    for prompt in PROMPTS:
        if prompt["prompt_id"] == prompt_id:
            return prompt["text"]
    raise KeyError(f"Unknown prompt_id: {prompt_id}")


if __name__ == "__main__":
    for i, prompt in enumerate(PROMPTS):
        print(f"{i:02d} | {prompt['prompt_id']} | {prompt['text']}")