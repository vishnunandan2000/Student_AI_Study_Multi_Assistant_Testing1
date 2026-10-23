import os
import sys
import uuid
import html
import shutil
import logging
import traceback
import urllib.parse
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
from PIL import Image
import gradio as gr

logger = logging.getLogger(__name__)

# Add root directory to sys.path to ensure backend modules can be imported seamlessly
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.core.llm import get_llm, check_ollama_connection
from backend.agents.orchestrator import app_graph
from backend.modules.audio_engine import generate_audio
from backend.modules.video_engine import generate_text_to_video, generate_3d_educational_video, generate_image_to_video, extract_image_path, StoryboardEngine
from backend.modules.image_engine import generate_ai_image, generate_ai_image_to_image
from backend.modules.translator import translate_text
from langchain_core.messages import HumanMessage, AIMessage
from backend.core import chat_history_db
from backend.modules.youtube_analyzer import (
    analyze_youtube_video,
    export_analysis_markdown,
    export_analysis_json,
    export_analysis_csv,
    validate_youtube_url
)
from backend.modules.video_editor_studio import (
    run_automatic_video_editing_pipeline,
    run_automatic_video_editing_generator,
    cancel_active_studio_job,
    render_stage_status_card,
    inspect_video_media,
    generate_youtube_thumbnails,
    generate_vertical_video_covers,
    generate_handwritten_study_notes,
    generate_engagement_animation_assets
)
from backend.modules.document_processor import (
    run_document_processing_generator,
    render_document_status_card,
    cancel_document_job
)
from backend.core import studio_db
from backend.core.moviepy_compat import log_startup_diagnostics
import pandas as pd
import plotly.graph_objects as go

# Log media processing environment & MoviePy diagnostics
log_startup_diagnostics()


def get_ollama_status_markdown() -> str:
    """Generates a health status banner for the local Ollama LLM server."""
    try:
        status = check_ollama_connection()
        if status.get('connected') and status.get('model_available'):
            model = status.get('models', ['llama3.2:latest'])[0]
            return f"🟢 **Ollama Server Online** | Model: `{model}`"
        elif status.get('connected'):
            return "🟡 **Ollama Server Online** | Model `llama3.2:latest` missing. Run `ollama pull llama3.2:latest`"
        else:
            return "🔴 **Ollama Server Offline** | Please start Ollama local server: `ollama serve`"
    except Exception as e:
        return f"🔴 **Ollama Status Unknown**: {e}"


def save_temp_file(content: str, filename: str) -> str:
    """Saves generated markdown text to a temporary file for downloading."""
    temp_dir = root_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_path = temp_dir / filename
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return str(file_path)


def extract_text_from_file(file_obj) -> str:
    """Extract text content from uploaded file objects (PDF, DOCX, PPTX, TXT)."""
    if file_obj is None:
        return ""
    
    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    file_name = file_path.lower()
    text = ""
    try:
        if file_name.endswith(".pdf"):
            import fitz
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text() + "\n"
        elif file_name.endswith(".docx"):
            import docx
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_name.endswith(".pptx"):
            import pptx
            prs = pptx.Presentation(file_path)
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        text = f"Error reading file {file_name}: {e}"
    return text.strip()


# ==========================================
# Event Handler Functions
# ==========================================

def get_session_choices() -> list:
    """Returns formatted list of choices for session selector dropdown."""
    sessions = chat_history_db.get_all_sessions()
    choices = [("➕ Start New Chat", "new")]
    for s in sessions:
        title = s["title"]
        if len(title) > 35:
            title = title[:35] + "..."
        created_str = s["created_at"].split("T")[0]
        label = f"💬 {title} ({created_str})"
        choices.append((label, s["session_id"]))
    return choices


def get_default_session_id():
    """Returns the ID of the most recent session or 'new' if none exist."""
    sessions = chat_history_db.get_all_sessions()
    if sessions:
        return sessions[0]["session_id"]
    return "new"


def get_initial_messages():
    """Retrieves messages for the default active session on page load."""
    default_id = get_default_session_id()
    if default_id and default_id != "new":
        return chat_history_db.get_session_messages(default_id)
    return []


def process_chat_attachments(file_objs, image_objs) -> tuple:
    """
    Processes uploaded documents and images for the AI Chatbot.
    Returns:
        (prompt_attachment_context: str, display_attachment_markdown: str)
    """
    prompt_parts = []
    display_parts = []
    
    uploads_dir = root_dir / "temp" / "chat_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # Process Document Files
    if file_objs:
        if not isinstance(file_objs, list):
            file_objs = [file_objs]
        for f_obj in file_objs:
            if f_obj is None:
                continue
            f_path = f_obj.name if hasattr(f_obj, "name") else str(f_obj)
            f_name = Path(f_path).name
            extracted_text = extract_text_from_file(f_path)
            if extracted_text:
                truncated = extracted_text[:15000]
                prompt_parts.append(f"📄 **[Attached Document: {f_name}]**\n```\n{truncated}\n```")
                display_parts.append(f"📄 **[Attached Document: {f_name}]**")

    # Process Image Files
    if image_objs:
        if not isinstance(image_objs, list):
            image_objs = [image_objs]
        for img_obj in image_objs:
            if img_obj is None:
                continue
            orig_path = img_obj.name if hasattr(img_obj, "name") else str(img_obj)
            img_path_obj = Path(orig_path)
            if not img_path_obj.exists():
                continue
            
            dest_name = f"{uuid.uuid4().hex[:8]}_{img_path_obj.name}"
            dest_path = uploads_dir / dest_name
            shutil.copy(str(img_path_obj), str(dest_path))
            
            # Inspect image details via PIL
            img_info = ""
            try:
                with Image.open(dest_path) as im:
                    img_info = f"Format: {im.format}, Size: {im.size[0]}x{im.size[1]}px, Mode: {im.mode}"
            except Exception:
                img_info = "Image loaded successfully."
                
            prompt_parts.append(f"🖼️ **[Attached Image: {img_path_obj.name}]** ({img_info})")
            
            # Form display markdown tag with file URL/path
            file_url = f"/gradio_api/file={dest_path.resolve().as_posix()}"
            display_parts.append(f"🖼️ **[Attached Image: {img_path_obj.name}]**\n![{img_path_obj.name}]({file_url})")

    prompt_context = "\n\n".join(prompt_parts)
    display_markdown = "\n\n".join(display_parts)
    return prompt_context, display_markdown


def update_mode_ui(mode: str) -> tuple:
    """Updates UI inputs based on the selected generation mode."""
    if mode == "🖼️ Text → Image":
        return gr.update(visible=False, value=None), gr.update(placeholder="Describe the image, visual diagram, or illustration to generate...")
    elif mode == "🖼️ Image → Image":
        return gr.update(visible=True), gr.update(placeholder="Optional instructions for modifying or annotating the reference image...")
    elif mode == "🎬 Text → Video":
        return gr.update(visible=False, value=None), gr.update(placeholder="Describe any video topic or concept (e.g., Photosynthesis in plants, The Solar System, How Airplanes Fly)...")
    elif mode == "🎬 Image → Video":
        return gr.update(visible=True), gr.update(placeholder="Optional motion prompt (e.g., 'zoom in', 'pan right', 'pan down', 'cinematic sweep', 'pulse')...")
    else: # 💬 Chat
        return gr.update(visible=False, value=None), gr.update(placeholder="Ask me anything or attach files/images...")


def chat_respond(user_message: str, chat_history: list, session_id: str, chatbot_mode: str = "💬 Chat", gen_ref_image=None, uploaded_files=None, uploaded_images=None) -> tuple:
    """Handles multi-turn conversation, Text->Image, Image->Image, Text->Video, and Image->Video generation with persistence via SQLite."""
    prompt_context, display_markdown = process_chat_attachments(uploaded_files, uploaded_images)
    full_user_text = user_message.strip() if user_message else ""
    
    if not full_user_text and not display_markdown and not gen_ref_image:
        current_msgs = chat_history_db.get_session_messages(session_id) if session_id and session_id != "new" else []
        choices = get_session_choices()
        return "", current_msgs, session_id, gr.update(choices=choices, value=session_id), None, None, None, None

    # Ensure a valid session exists
    if not session_id or session_id == "new":
        title_source = full_user_text or "Attached Media/Prompt"
        title = f"[{chatbot_mode}] " + title_source[:30] + ("..." if len(title_source) > 30 else "")
        session_id = chat_history_db.create_session(title=title)

    gen_dir = root_dir / "temp" / "generated_media"
    gen_dir.mkdir(parents=True, exist_ok=True)
    generated_media_path = None

    # -------------------------------------------------------------
    # MODE 1: TEXT -> IMAGE (PNG Format)
    # -------------------------------------------------------------
    if chatbot_mode == "🖼️ Text → Image":
        user_prompt = full_user_text or "Educational Illustration"
        saved_user_message = f"🖼️ **[Text → Image Prompt]**: {user_prompt}"
        
        try:
            out_img_path = gen_dir / f"txt2img_{uuid.uuid4().hex[:8]}.png"
            final_img_path = generate_ai_image(user_prompt, str(out_img_path))
            img_path_obj = Path(final_img_path).resolve()
            img_url = f"/gradio_api/file={img_path_obj.as_posix()}"
            generated_media_path = str(img_path_obj)
            
            assistant_response = (
                f'<div style="font-family: system-ui, -apple-system, sans-serif; padding: 4px;">'
                f'<p style="margin: 0 0 10px 0; font-size: 15px; font-weight: 600; color: #e2e8f0;">🖼️ <strong>Prompt:</strong> <em>"{user_prompt}"</em></p>'
                f'<div style="margin-bottom: 12px; border-radius: 12px; overflow: hidden; border: 1px solid #334155; box-shadow: 0 4px 12px rgba(0,0,0,0.3); background: #0f172a;">'
                f'<img src="{img_url}" alt="{user_prompt}" style="width: 100%; height: auto; display: block; max-height: 450px; object-fit: contain; background: #0f172a;" />'
                f'</div>'
                f'<div>'
                f'<a href="{img_url}" download="{img_path_obj.name}" target="_blank" style="display: inline-flex; align-items: center; gap: 8px; padding: 9px 18px; background: linear-gradient(135deg, #2563eb, #1d4ed8); color: #ffffff; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 14px; box-shadow: 0 2px 6px rgba(37,99,235,0.4); transition: all 0.2s ease;">'
                f'📥 Download Image (PNG)'
                f'</a>'
                f'</div>'
                f'</div>'
            )
        except Exception as e:
            assistant_response = f"❌ **Text → Image Generation Error**: {e}"

    # -------------------------------------------------------------
    # MODE 2: IMAGE -> IMAGE (PNG Format)
    # -------------------------------------------------------------
    elif chatbot_mode == "🖼️ Image → Image":
        user_prompt = full_user_text or "Enhanced Image Transformation"
        ref_img_path = gen_ref_image if gen_ref_image else (uploaded_images[0] if uploaded_images else None)
        
        if not ref_img_path:
            assistant_response = "⚠️ **Missing Reference Image**: Please upload a reference image above to perform Image → Image transformation."
            saved_user_message = f"🖼️ **[Image → Image Request]**: {user_prompt} *(No image uploaded)*"
        else:
            ref_path_str = ref_img_path.name if hasattr(ref_img_path, "name") else str(ref_img_path)
            ref_path_obj = Path(ref_path_str).resolve()
            ref_url = f"/gradio_api/file={ref_path_obj.as_posix()}"
            saved_user_message = f"🖼️ **[Image → Image Request]**\n![Reference Image]({ref_url})\n\nInstructions: {user_prompt}"
            try:
                out_img_path = gen_dir / f"img2img_{uuid.uuid4().hex[:8]}.png"
                final_img_path = generate_ai_image_to_image(ref_path_str, user_prompt, str(out_img_path))
                img_path_obj = Path(final_img_path).resolve()
                img_url = f"/gradio_api/file={img_path_obj.as_posix()}"
                generated_media_path = str(img_path_obj)
                
                assistant_response = (
                    f'<div style="font-family: system-ui, -apple-system, sans-serif; padding: 4px;">'
                    f'<p style="margin: 0 0 10px 0; font-size: 15px; font-weight: 600; color: #e2e8f0;">🖼️ <strong>Transformation:</strong> <em>"{user_prompt}"</em></p>'
                    f'<div style="display: flex; gap: 14px; margin-bottom: 14px; flex-wrap: wrap;">'
                    f'<div style="flex: 1; min-width: 260px; border-radius: 12px; overflow: hidden; border: 1px solid #334155; box-shadow: 0 4px 12px rgba(0,0,0,0.3); background: #0f172a;">'
                    f'<div style="background: #1e293b; padding: 6px 12px; font-size: 13px; font-weight: 600; color: #94a3b8;">Original Reference</div>'
                    f'<img src="{ref_url}" alt="Original Reference" style="width: 100%; height: auto; display: block; max-height: 380px; object-fit: contain; background: #0f172a;" />'
                    f'</div>'
                    f'<div style="flex: 1; min-width: 260px; border-radius: 12px; overflow: hidden; border: 1px solid #3b82f6; box-shadow: 0 4px 12px rgba(59,130,246,0.3); background: #0f172a;">'
                    f'<div style="background: #1d4ed8; padding: 6px 12px; font-size: 13px; font-weight: 600; color: #ffffff;">✨ Transformed Output</div>'
                    f'<img src="{img_url}" alt="{user_prompt}" style="width: 100%; height: auto; display: block; max-height: 380px; object-fit: contain; background: #0f172a;" />'
                    f'</div>'
                    f'</div>'
                    f'<div>'
                    f'<a href="{img_url}" download="{img_path_obj.name}" target="_blank" style="display: inline-flex; align-items: center; gap: 8px; padding: 9px 18px; background: linear-gradient(135deg, #2563eb, #1d4ed8); color: #ffffff; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 14px; box-shadow: 0 2px 6px rgba(37,99,235,0.4); transition: all 0.2s ease;">'
                    f'📥 Download Transformed Image (PNG)'
                    f'</a>'
                    f'</div>'
                    f'</div>'
                )
            except Exception as e:
                assistant_response = f"❌ **Image → Image Generation Error**: {e}"

    # -------------------------------------------------------------
    # MODE 3: TEXT -> VIDEO (MP4 Format)
    # -------------------------------------------------------------
    elif chatbot_mode == "🎬 Text → Video":
        user_prompt = full_user_text or "Educational Explainer Video"
        saved_user_message = f"🎬 **[Text → Video Prompt]**: {user_prompt}"
        
        try:
            out_vid_path = gen_dir / f"txt2vid_{uuid.uuid4().hex[:8]}.mp4"
            final_vid_path = generate_text_to_video(
                prompt=user_prompt,
                output_path=str(out_vid_path),
                language="Auto-Detect",
                persona="Teacher Female",
                resolution="720p",
                num_scenes=3
            )
            vid_path_obj = Path(final_vid_path).resolve()
            vid_url = f"/gradio_api/file={vid_path_obj.as_posix()}"
            generated_media_path = str(vid_path_obj)
            
            assistant_response = (
                f'<div style="font-family: system-ui, -apple-system, sans-serif; padding: 4px;">'
                f'<p style="margin: 0 0 10px 0; font-size: 15px; font-weight: 600; color: #e2e8f0;">🎬 <strong>Video Topic:</strong> <em>"{user_prompt}"</em></p>'
                f'<div style="margin-bottom: 12px; border-radius: 12px; overflow: hidden; border: 1px solid #334155; box-shadow: 0 4px 12px rgba(0,0,0,0.3); background: #0f172a;">'
                f'<video controls src="{vid_url}" width="100%" style="max-height: 420px; display: block; width: 100%; border-radius: 12px;"></video>'
                f'</div>'
                f'<div>'
                f'<a href="{vid_url}" download="{vid_path_obj.name}" target="_blank" style="display: inline-flex; align-items: center; gap: 8px; padding: 9px 18px; background: linear-gradient(135deg, #059669, #10b981); color: #ffffff; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 14px; box-shadow: 0 2px 6px rgba(16,185,129,0.4); transition: all 0.2s ease;">'
                f'🎥 Download Video (MP4)'
                f'</a>'
                f'</div>'
                f'</div>'
            )
        except Exception as e:
            assistant_response = f"❌ **Text → Video Generation Error**: {e}"

    # -------------------------------------------------------------
    # MODE 4: IMAGE -> VIDEO (MP4 Format)
    # -------------------------------------------------------------
    elif chatbot_mode == "🎬 Image → Video":
        user_prompt = full_user_text or "Educational Image Animation & Narration"
        
        # Robustly locate reference image from all potential input sources
        candidate_inputs = [gen_ref_image]
        if uploaded_images:
            if isinstance(uploaded_images, list):
                candidate_inputs.extend(uploaded_images)
            else:
                candidate_inputs.append(uploaded_images)
        if uploaded_files:
            if isinstance(uploaded_files, list):
                candidate_inputs.extend(uploaded_files)
            else:
                candidate_inputs.append(uploaded_files)

        valid_ref_path = None
        for cand in candidate_inputs:
            p = extract_image_path(cand)
            if p:
                valid_ref_path = p
                break
        
        if not valid_ref_path:
            assistant_response = (
                "⚠️ **Missing Reference Image**: Please upload an image using the "
                "**Upload Reference Image** box above (or attach an image file) to generate an Image → Video animation."
            )
            saved_user_message = f"🎬 **[Image → Video Request]**: {user_prompt} *(No image uploaded)*"
        else:
            ref_path_obj = Path(valid_ref_path).resolve()
            ref_url = f"/gradio_api/file={ref_path_obj.as_posix()}"
            saved_user_message = f"🎬 **[Image → Video Request]**\n![Reference Image]({ref_url})\n\nPrompt: {user_prompt}"
            try:
                out_vid_path = gen_dir / f"img2vid_{uuid.uuid4().hex[:8]}.mp4"
                final_vid_path = generate_image_to_video(valid_ref_path, user_prompt, str(out_vid_path))
                vid_path_obj = Path(final_vid_path).resolve()
                vid_url = f"/gradio_api/file={vid_path_obj.as_posix()}"
                generated_media_path = str(vid_path_obj)
                
                assistant_response = (
                    f'<div style="font-family: system-ui, -apple-system, sans-serif; padding: 4px;">'
                    f'<p style="margin: 0 0 10px 0; font-size: 15px; font-weight: 600; color: #e2e8f0;">🎬 <strong>Animated Video:</strong> <em>"{user_prompt}"</em></p>'
                    f'<div style="display: flex; gap: 14px; margin-bottom: 14px; flex-wrap: wrap;">'
                    f'<div style="flex: 1; min-width: 260px; border-radius: 12px; overflow: hidden; border: 1px solid #334155; box-shadow: 0 4px 12px rgba(0,0,0,0.3); background: #0f172a;">'
                    f'<div style="background: #1e293b; padding: 6px 12px; font-size: 13px; font-weight: 600; color: #94a3b8;">📷 Original Reference Image</div>'
                    f'<img src="{ref_url}" alt="Original Reference" style="width: 100%; height: auto; display: block; max-height: 380px; object-fit: contain; background: #0f172a;" />'
                    f'</div>'
                    f'<div style="flex: 1; min-width: 260px; border-radius: 12px; overflow: hidden; border: 1px solid #10b981; box-shadow: 0 4px 12px rgba(16,185,129,0.3); background: #0f172a;">'
                    f'<div style="background: #059669; padding: 6px 12px; font-size: 13px; font-weight: 600; color: #ffffff;">✨ Rendered Video Animation (MP4)</div>'
                    f'<video controls autoplay loop muted src="{vid_url}" style="width: 100%; height: auto; display: block; max-height: 380px; object-fit: contain; background: #0f172a;"></video>'
                    f'</div>'
                    f'</div>'
                    f'<div>'
                    f'<a href="{vid_url}" download="{vid_path_obj.name}" target="_blank" style="display: inline-flex; align-items: center; gap: 8px; padding: 9px 18px; background: linear-gradient(135deg, #059669, #10b981); color: #ffffff; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 14px; box-shadow: 0 2px 6px rgba(16,185,129,0.4); transition: all 0.2s ease;">'
                    f'🎥 Download Video (MP4)'
                    f'</a>'
                    f'</div>'
                    f'</div>'
                )
            except Exception as e:
                assistant_response = f"❌ **Image → Video Generation Error**: {e}"

    # -------------------------------------------------------------
    # MODE 5: CHAT (DEFAULT)
    # -------------------------------------------------------------
    else:
        # Construct displayed user message content for UI & DB history
        display_content_parts = []
        if display_markdown:
            display_content_parts.append(display_markdown)
        if full_user_text:
            display_content_parts.append(full_user_text)
        saved_user_message = "\n\n".join(display_content_parts)

        # Construct prompt message for LangGraph AI agent
        agent_prompt_parts = []
        if prompt_context:
            agent_prompt_parts.append(prompt_context)
        if full_user_text:
            agent_prompt_parts.append(full_user_text)
        prompt_for_agent = "\n\n".join(agent_prompt_parts)

        # Reconstruct past messages from DB for context
        past_db_messages = chat_history_db.get_session_messages(session_id)
        messages = []
        for item in past_db_messages:
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=prompt_for_agent))

        # Invoke Agent Graph
        try:
            inputs = {"messages": messages}
            result = app_graph.invoke(inputs)
            assistant_response = result['messages'][-1].content
        except Exception as e:
            assistant_response = f"❌ Error executing AI Agent: {e}\n\nPlease ensure Ollama is running (`ollama serve`) and `llama3.2:latest` is pulled."

    # Persist user prompt and assistant response into SQLite
    chat_history_db.add_message(session_id, "user", saved_user_message)
    chat_history_db.add_message(session_id, "assistant", assistant_response)

    # Fetch updated messages from DB
    updated_history = chat_history_db.get_session_messages(session_id)
    choices = get_session_choices()

    return "", updated_history, session_id, gr.update(choices=choices, value=session_id), None, None, None, gr.update(value=generated_media_path, visible=bool(generated_media_path))


def select_session(session_id: str) -> tuple:
    """Loads selected session's chat history."""
    if not session_id or session_id == "new":
        return [], "new"
    messages = chat_history_db.get_session_messages(session_id)
    return messages, session_id


def start_new_chat() -> tuple:
    """Resets UI to a fresh new chat session state."""
    choices = get_session_choices()
    return [], "new", gr.update(choices=choices, value="new"), None, None, None, None


def delete_selected_chat(session_id: str) -> tuple:
    """Deletes the active session and selects the next available session."""
    if session_id and session_id != "new":
        chat_history_db.delete_session(session_id)
    
    sessions = chat_history_db.get_all_sessions()
    choices = get_session_choices()
    
    if sessions:
        new_active_id = sessions[0]["session_id"]
        messages = chat_history_db.get_session_messages(new_active_id)
        return messages, new_active_id, gr.update(choices=choices, value=new_active_id), None, None, None, None
    else:
        return [], "new", gr.update(choices=choices, value="new"), None, None, None, None


def delete_all_chat_history() -> tuple:
    """Permanently deletes all stored chat sessions."""
    chat_history_db.delete_all_sessions()
    choices = get_session_choices()
    return [], "new", gr.update(choices=choices, value="new"), None, None, None, None


def generate_study_plan(topic: str, level: str, duration: str, hours_per_day: float, goals: str) -> tuple:
    if not topic.strip():
        return "⚠️ Please enter a subject or topic to generate a study plan.", None
    
    prompt = (
        f"Act as an expert AI Study Planner. Create a detailed, highly structured {duration} study plan for the topic: '{topic}'.\n"
        f"- Current Level: {level}\n"
        f"- Time Commitment: {int(hours_per_day)} hours per day\n"
        f"- Specific Goals: {goals if goals.strip() else 'Understand core concepts thoroughly and build a solid foundation.'}\n\n"
        "Please provide a day-by-day (or week-by-week) breakdown. Include specific topics, suggested study techniques, and clear milestones. Format beautifully in Markdown."
    )
    
    try:
        inputs = {"messages": [HumanMessage(content=prompt)]}
        result = app_graph.invoke(inputs)
        content = result['messages'][-1].content
        
        file_slug = topic.replace(" ", "_").lower()
        file_path = save_temp_file(content, f"{file_slug}_study_plan.md")
        return content, file_path
    except Exception as e:
        return f"❌ Error generating study plan: {e}", None


def generate_exam_material(topic: str, prep_type: str, difficulty: str, num_questions: float, additional_context: str) -> tuple:
    if not topic.strip():
        return "⚠️ Please enter a subject or topic to proceed.", None
    
    prompt = (
        f"Act as an expert AI Exam Coach. Generate a '{prep_type}' for the topic: '{topic}'.\n"
        f"- Difficulty Level: {difficulty}\n"
        f"- Quantity: {int(num_questions)} questions/items\n"
        f"- Additional Context: {additional_context if additional_context.strip() else 'None'}\n\n"
        "If generating a test or quiz, provide questions first and include the answer key at the end. Format beautifully in Markdown."
    )
    
    try:
        inputs = {"messages": [HumanMessage(content=prompt)]}
        result = app_graph.invoke(inputs)
        content = result['messages'][-1].content
        
        file_slug = f"{topic.replace(' ', '_').lower()}_{prep_type.replace(' ', '_').lower()}"
        file_path = save_temp_file(content, f"{file_slug}.md")
        return content, file_path
    except Exception as e:
        return f"❌ Error generating exam material: {e}", None


def run_coding_lab(mode: str, language: str, code_snippet: str, error_msg: str, dsa_topic: str) -> tuple:
    if mode in ["Debug Code", "Explain Code", "Optimize Code"] and not code_snippet.strip():
        return "⚠️ Please paste some code to proceed.", None
    if mode == "Learn a Topic (e.g., DSA)" and not dsa_topic.strip():
        return "⚠️ Please enter a topic to learn.", None
        
    if mode == "Debug Code":
        prompt = f"Act as a Senior Coding Mentor. Debug this {language} code:\n\n```{language}\n{code_snippet}\n```\n\nError/Issue: {error_msg}\n\nExplain the bug and provide corrected code."
    elif mode == "Explain Code":
        prompt = f"Act as a Senior Coding Mentor. Explain this {language} code step-by-step:\n\n```{language}\n{code_snippet}\n```"
    elif mode == "Optimize Code":
        prompt = f"Act as a Senior Coding Mentor. Optimize this {language} code for performance and readability:\n\n```{language}\n{code_snippet}\n```"
    else:
        prompt = f"Act as a Senior Coding Mentor. Teach '{dsa_topic}' in {language} with explanations, analogies, and practical code."
        
    try:
        inputs = {"messages": [HumanMessage(content=prompt)]}
        result = app_graph.invoke(inputs)
        content = result['messages'][-1].content
        
        file_slug = f"coding_lab_{mode.replace(' ', '_').lower()}"
        file_path = save_temp_file(content, f"{file_slug}.md")
        return content, file_path
    except Exception as e:
        return f"❌ Error during coding analysis: {e}", None


def generate_project_plan(project_idea: str, tech_stack: str, generate_type: str, experience_level: str) -> tuple:
    if not project_idea.strip():
        return "⚠️ Please enter a project idea to proceed.", None
        
    stack_context = f"using {tech_stack}" if tech_stack.strip() else "using an optimal tech stack"
    prompt = (
        f"Act as a Senior Software Architect. The user is a {experience_level} developer.\n"
        f"Project Idea: {project_idea}\n"
        f"Tech Stack: {stack_context}\n\n"
        f"Generate a detailed '{generate_type}' for this project formatted beautifully in Markdown."
    )
    
    try:
        inputs = {"messages": [HumanMessage(content=prompt)]}
        result = app_graph.invoke(inputs)
        content = result['messages'][-1].content
        
        file_slug = f"{project_idea.replace(' ', '_').lower()}_{generate_type.replace(' ', '_').lower()}"
        file_path = save_temp_file(content, f"{file_slug}.md")
        return content, file_path
    except Exception as e:
        return f"❌ Error generating project plan: {e}", None


def analyze_document(doc_file, pasted_text: str, task_type: str, specific_question: str) -> tuple:
    document_text = ""
    if doc_file is not None:
        document_text = extract_text_from_file(doc_file)
    elif pasted_text.strip():
        document_text = pasted_text.strip()
        
    if not document_text.strip():
        return "⚠️ Please upload a document or paste text to analyze.", None
    if task_type == "Ask a Specific Question" and not specific_question.strip():
        return "⚠️ Please enter a question about the document.", None
        
    max_chars = 25000
    truncated_text = document_text[:max_chars]
    
    if task_type == "Summarize Document":
        prompt = f"Act as an expert Document Analysis Specialist. Provide a comprehensive summary of:\n\n{truncated_text}"
    elif task_type == "Extract Key Takeaways":
        prompt = f"Act as an expert Document Analysis Specialist. Extract key takeaways from:\n\n{truncated_text}"
    else:
        prompt = f"Act as an expert Document Analysis Specialist. Answer the question: '{specific_question}' based ONLY on context:\n\n{truncated_text}"
        
    try:
        inputs = {"messages": [HumanMessage(content=prompt)]}
        result = app_graph.invoke(inputs)
        content = result['messages'][-1].content
        
        file_slug = f"doc_analysis_{task_type.replace(' ', '_').lower()}"
        file_path = save_temp_file(content, f"{file_slug}.md")
        return content, file_path
    except Exception as e:
        return f"❌ Error analyzing document: {e}", None


# =========================================================================
# Document Processing Real-Time Status & Progress Handlers
# =========================================================================

def get_document_queue_dataframe() -> pd.DataFrame:
    """Retrieves recent document processing jobs for interactive UI queue monitoring."""
    jobs = studio_db.list_document_jobs(limit=15)
    if not jobs:
        return pd.DataFrame(columns=["Job ID", "Document / Task", "Version", "Status", "Progress", "Updated"])
    rows = []
    for j in jobs:
        params = j.get("input_params", {})
        fname = params.get("filename", "Document.txt")
        ver = f"v{params.get('version', 1)}"
        rows.append({
            "Job ID": j["job_id"],
            "Document / Task": f"{fname} ({params.get('task_type', 'Analysis')[:24]})",
            "Version": ver,
            "Status": j["status"].upper(),
            "Progress": f"{j['progress']:.0f}%",
            "Updated": j["updated_at"].split("T")[1][:8] if "T" in j["updated_at"] else j["updated_at"]
        })
    return pd.DataFrame(rows)


def run_document_intelligence_ui(
    doc_file,
    doc_text: str,
    task_type: str,
    specific_question: str,
    comparison_file=None,
    existing_doc_id: str = "",
    progress=gr.Progress(track_tqdm=False)
):
    if doc_file is None and not (doc_text and doc_text.strip()):
        empty_card = render_document_status_card(
            1, 8, "Upload Validation", 0.0, 0.0, "failed",
            error_msg="Please upload a document (.pdf, .docx, .pptx, .txt) or enter text to process."
        )
        yield (
            empty_card,
            "⚠️ **Upload required:** Please upload a document file or paste content to begin processing.",
            None, None,
            get_document_queue_dataframe(),
            "⚠️ No document file or text was provided.",
            "",
            gr.update(interactive=True),
            gr.update(interactive=False)
        )
        return

    job_id = f"job_doc_{uuid.uuid4().hex[:8]}"
    doc_id = existing_doc_id.strip() if existing_doc_id and existing_doc_id.strip() else f"doc_{uuid.uuid4().hex[:8]}"

    gen = run_document_processing_generator(
        file_obj=doc_file,
        pasted_text=doc_text.strip() if doc_text else "",
        task_type=task_type,
        specific_question=specific_question.strip() if specific_question else "",
        comparison_file_obj=comparison_file,
        existing_doc_id=doc_id,
        job_id=job_id
    )

    for state in gen:
        pct = max(0.0, min(1.0, state["progress_percent"] / 100.0))
        stage_name = state["stage_name"]
        stage_idx = state["stage_index"]
        progress(pct, desc=f"Stage {stage_idx}/8: {stage_name}")

        status_html = state["status_card_html"]
        logs = state["log_text"]
        res = state.get("results", {})

        if state["is_complete"]:
            yield (
                status_html,
                res.get("analysis_markdown", "### Analysis Complete"),
                res.get("report_md_path"),
                res.get("report_json_path"),
                get_document_queue_dataframe(),
                logs,
                job_id,
                gr.update(interactive=True),
                gr.update(interactive=False)
            )
        elif state["is_error"] or state["is_cancelled"]:
            yield (
                status_html,
                f"### ❌ Process {state['status_label'].capitalize()}\n{logs}",
                None, None,
                get_document_queue_dataframe(),
                logs,
                job_id,
                gr.update(interactive=True),
                gr.update(interactive=False)
            )
        else:
            yield (
                status_html,
                "### 🔄 Document Processing in Progress...\nPlease observe real-time stage checklist above.",
                None, None,
                get_document_queue_dataframe(),
                logs,
                job_id,
                gr.update(interactive=False),
                gr.update(interactive=True)
            )


def cancel_active_document_job_ui(active_job_id: str):
    if not active_job_id or not active_job_id.strip():
        return "⚠️ No active document processing job is currently running.", gr.update(interactive=False)
    success, msg = cancel_document_job(active_job_id.strip())
    return f"🛑 {msg}", gr.update(interactive=False)


def run_audio_generation(text_to_convert: str) -> tuple:
    if not text_to_convert.strip():
        return None, None, "⚠️ Please enter some text."
        
    try:
        temp_dir = root_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_file = temp_dir / "output_audio.mp3"
        
        final_path = generate_audio(text_to_convert, str(output_file))
        return final_path, final_path, "✅ Audio generated successfully!"
    except Exception as e:
        return None, None, f"❌ Audio Generation Error: {e}"


def run_video_generation(input_mode: str, script_text: str, doc_file, language: str, persona: str, theme: str, prop_type: str, resolution: str, aspect_ratio: str) -> tuple:
    text = ""
    if input_mode == "Upload Document (PDF/DOCX/PPTX/TXT)" and doc_file is not None:
        text = extract_text_from_file(doc_file)
    else:
        text = script_text.strip()
        
    if not text:
        return None, None, None, "⚠️ Please enter a script or upload an educational document."
        
    try:
        temp_dir = root_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        video_output = temp_dir / "professor_robocat_multilingual.mp4"
        
        res_val = resolution.split()[0] if resolution else "1080p"
        asp_val = aspect_ratio.split()[0] if aspect_ratio else "16:9"
        
        final_video_path = generate_3d_educational_video(
            text=text,
            output_path=str(video_output),
            language=language,
            persona=persona,
            resolution=res_val,
            aspect_ratio=asp_val,
            theme=theme,
            prop_type=prop_type
        )
        
        audio_file_p = str(temp_dir / "narration_speech.mp3")
        srt_file_p = str(temp_dir / "subtitles.srt")
        
        audio_res = audio_file_p if Path(audio_file_p).exists() else None
        srt_res = srt_file_p if Path(srt_file_p).exists() else None
        
        return final_video_path, audio_res, srt_res, "🎉 3D Multilingual Educational Video successfully rendered!"
    except Exception as e:
        return None, None, None, f"❌ Video Generation Error: {e}"


def preview_storyboard(script_text: str) -> str:
    if not script_text.strip():
        return "Enter a script above to preview the storyboard."
    scenes = StoryboardEngine.create_scenes(script_text)
    out = "### 📋 Auto-Generated Storyboard Breakdown\n\n"
    for sc in scenes:
        out += f"**Scene {sc['scene_num']}: {sc['title']}**\n- **Pose:** `{sc['pose']}` | **Expression:** `{sc['expression']}`\n- **Script:** *\"{sc['script']}\"*\n\n"
    return out


# ==========================================
# Module A & B UI Handlers
# ==========================================

def render_comments_explorer_view(
    comments_pool: list,
    total_video_comments: Optional[int] = None,
    sentiment_filter: str = "All Sentiments",
    search_query: str = "",
    sort_by: str = "Default (Retrieved)",
    page: int = 1,
    page_size: str = "10",
    visible_limit: int = 10,
    use_load_more: bool = False
):
    """
    Renders rich interactive HTML cards, summary banner, page indicator, and pandas DataFrame
    for the Audience Comments Explorer.
    """
    empty_df = pd.DataFrame(columns=["#", "Author", "Sentiment", "Likes", "Published At", "Reply?", "Comment"])
    if not comments_pool:
        empty_summary = """
        <div style="background: #0f172a; border: 1px dashed #334155; border-radius: 12px; padding: 22px; text-align: center; color: #94a3b8;">
            <div style="font-size: 26px; margin-bottom: 6px;">💬</div>
            <strong style="color: #cbd5e1; font-size: 15px;">No comments loaded yet.</strong>
            <p style="margin: 4px 0 0 0; font-size: 13px;">Enter a YouTube URL above, choose your comment retrieval window (e.g. 10, 25, 50, 100, or All Available Comments), then click <strong>Analyze Video 🚀</strong>.</p>
        </div>
        """
        empty_feed = """<div style="text-align: center; padding: 25px; color: #64748b;">No comments available.</div>"""
        return empty_summary, empty_feed, empty_df, "Showing 0 comments", 1, 10

    # 1. Compute Pool Counts
    total_pool = len(comments_pool)
    pos_count = sum(1 for c in comments_pool if c.get("category") == "Positive")
    neu_count = sum(1 for c in comments_pool if c.get("category") == "Neutral")
    neg_count = sum(1 for c in comments_pool if c.get("category") == "Negative")

    # 2. Filter by Sentiment
    filtered = list(comments_pool)
    if "positive" in sentiment_filter.lower():
        filtered = [c for c in filtered if c.get("category") == "Positive"]
    elif "neutral" in sentiment_filter.lower():
        filtered = [c for c in filtered if c.get("category") == "Neutral"]
    elif "negative" in sentiment_filter.lower():
        filtered = [c for c in filtered if c.get("category") == "Negative"]

    # 3. Filter by Search Query
    q = search_query.strip().lower()
    if q:
        filtered = [
            c for c in filtered
            if q in c.get("text", "").lower() or q in c.get("author", "").lower()
        ]

    # 4. Sort Comments
    if "liked" in sort_by.lower():
        filtered = sorted(filtered, key=lambda c: int(c.get("likes", 0)), reverse=True)
    elif "newest" in sort_by.lower():
        filtered = sorted(filtered, key=lambda c: str(c.get("published_at", "")), reverse=True)
    elif "oldest" in sort_by.lower():
        filtered = sorted(filtered, key=lambda c: str(c.get("published_at", "")), reverse=False)

    total_filtered = len(filtered)

    # 5. Pagination / Load More Slicing
    if use_load_more:
        curr_vis = max(10, visible_limit)
        curr_vis = min(curr_vis, total_filtered) if total_filtered > 0 else 10
        displayed = filtered[:curr_vis]
        page = 1
        per_page = curr_vis
        total_pages = 1
    else:
        if page_size.lower() == "all":
            per_page = max(1, total_filtered)
            total_pages = 1
            page = 1
            displayed = filtered
            curr_vis = len(displayed)
        else:
            try:
                per_page = max(1, int(page_size))
            except Exception:
                per_page = 10
            total_pages = max(1, (total_filtered + per_page - 1) // per_page)
            page = max(1, min(page, total_pages))
            start_idx = (page - 1) * per_page
            end_idx = min(start_idx + per_page, total_filtered)
            displayed = filtered[start_idx:end_idx]
            curr_vis = len(displayed)

    # 6. Build Summary Banner HTML
    total_on_yt_badge = ""
    if total_video_comments is not None:
        total_on_yt_badge = f"""<span style="background: #1e293b; color: #94a3b8; border: 1px solid #334155; font-size: 12px; padding: 3px 9px; border-radius: 12px; font-weight: 500;">Total on YouTube: <strong>{total_video_comments:,}</strong></span>"""

    summary_html = f"""
    <div style="background: #0f172a; border: 1px solid #334155; border-radius: 12px; padding: 14px 18px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                <h4 style="margin: 0; color: #f8fafc; font-size: 16px; font-weight: bold;">💬 Audience Comments Explorer</h4>
                <span style="background: #2563eb; color: #fff; font-size: 12px; padding: 3px 10px; border-radius: 12px; font-weight: bold;">{total_pool} Comments Retrieved</span>
                {total_on_yt_badge}
            </div>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 12.5px;">All retrieved comments are analyzed and preserved without artificial truncation. Filter, search, and paginate below.</p>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: bold;">🟢 {pos_count} Positive</span>
            <span style="background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.3); padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: bold;">⚪ {neu_count} Neutral</span>
            <span style="background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: bold;">🔴 {neg_count} Negative</span>
        </div>
    </div>
    """

    # 7. Build Feed HTML Cards
    if not displayed:
        feed_html = """
        <div style="text-align: center; padding: 35px; color: #94a3b8; background: #0f172a; border-radius: 12px; border: 1px dashed #334155;">
            🔍 No comments found matching current filter or search criteria.
        </div>
        """
    else:
        cards_html_list = []
        for c in displayed:
            author = html.escape(c.get("author") or "Anonymous")
            initial = (author[0] if author else "U").upper()
            avatar_url = c.get("author_avatar") or ""

            pub_date = c.get("published_at", "")
            if len(pub_date) >= 10:
                date_display = pub_date[:10]
                if len(pub_date) >= 16:
                    date_display += " " + pub_date[11:16]
            else:
                date_display = pub_date or "Recent"

            if c.get("is_reply"):
                reply_badge = """<span style="background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">↪️ Reply</span>"""
            elif c.get("reply_count", 0) > 0:
                rc = c.get("reply_count", 0)
                reply_badge = f"""<span style="background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.3); padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">💬 {rc} {'replies' if rc > 1 else 'reply'}</span>"""
            else:
                reply_badge = """<span style="background: rgba(100, 116, 139, 0.15); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.25); padding: 2px 7px; border-radius: 4px; font-size: 11px;">Thread</span>"""

            cat = c.get("category", "Neutral")
            score = c.get("score", 0.0)
            if cat == "Positive":
                score_str = f"+{score:.1f}" if score > 0 else f"{score:.1f}"
                sentiment_badge = f"""<span style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: bold;">🟢 Positive ({score_str})</span>"""
            elif cat == "Negative":
                score_str = f"{score:.1f}"
                sentiment_badge = f"""<span style="background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: bold;">🔴 Negative ({score_str})</span>"""
            else:
                sentiment_badge = """<span style="background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.35); padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: bold;">⚪ Neutral</span>"""

            if avatar_url:
                avatar_img_html = f"""<img src="{avatar_url}" style="width: 36px; height: 36px; border-radius: 50%; object-fit: cover;" onerror="this.onerror=null; this.src='https://api.dicebear.com/7.x/initials/svg?seed={urllib.parse.quote(author)}';" />"""
            else:
                avatar_img_html = f"""<div style="width: 36px; height: 36px; border-radius: 50%; background: #3b82f6; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px;">{initial}</div>"""

            likes = c.get("likes", 0)
            likes_html = f"""<span style="background: #0f172a; border: 1px solid #334155; padding: 2px 8px; border-radius: 6px; color: #38bdf8; font-size: 11px; font-weight: bold;">👍 {likes}</span>"""
            text_escaped = html.escape(c.get("text", "")).replace("\n", "<br/>")

            card = f"""
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.25);">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        {avatar_img_html}
                        <div>
                            <div style="font-weight: 600; color: #f8fafc; font-size: 13.5px;">{author}</div>
                            <div style="font-size: 11px; color: #94a3b8;">{date_display}</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                        {reply_badge}
                        {sentiment_badge}
                        {likes_html}
                    </div>
                </div>
                <div style="margin: 0; color: #e2e8f0; font-size: 13.5px; line-height: 1.55; white-space: pre-wrap; word-break: break-word;">{text_escaped}</div>
            </div>
            """
            cards_html_list.append(card)

        feed_html = f"""<div style="max-height: 600px; overflow-y: auto; padding-right: 4px;">{''.join(cards_html_list)}</div>"""

    # 8. Build Tabular DataFrame
    df_rows = []
    for idx, c in enumerate(filtered, 1):
        df_rows.append({
            "#": idx,
            "Author": c.get("author", "Anonymous"),
            "Sentiment": c.get("category", "Neutral"),
            "Likes": int(c.get("likes", 0)),
            "Published At": str(c.get("published_at", "")),
            "Reply?": "Yes" if c.get("is_reply") else "No",
            "Comment": c.get("text", "")
        })
    df = pd.DataFrame(df_rows) if df_rows else empty_df

    # 9. Page Indicator
    if use_load_more:
        page_indicator = f"📄 **Showing {len(displayed)} of {total_filtered} matching comments** (Pool: {total_pool} retrieved)"
    else:
        page_indicator = f"📄 **Page {page} of {total_pages}** • Showing {len(displayed)} of {total_filtered} matching comments (Pool: {total_pool} retrieved)"

    return summary_html, feed_html, df, page_indicator, total_pages, curr_vis


def on_comments_filter_change(
    comments_pool: list,
    sentiment_filter: str,
    search_query: str,
    sort_by: str,
    page_size: str,
    current_page: int,
    visible_limit: int,
    total_comments: int = 0
):
    _, feed_html, df, indicator, total_pages, new_vis = render_comments_explorer_view(
        comments_pool=comments_pool or [],
        total_video_comments=total_comments,
        sentiment_filter=sentiment_filter,
        search_query=search_query,
        sort_by=sort_by,
        page=1,
        page_size=page_size,
        visible_limit=10,
        use_load_more=False
    )
    return feed_html, df, indicator, 1, new_vis


def on_comments_prev_page(
    comments_pool: list,
    sentiment_filter: str,
    search_query: str,
    sort_by: str,
    page_size: str,
    current_page: int,
    visible_limit: int,
    total_comments: int = 0
):
    new_page = max(1, (current_page or 1) - 1)
    _, feed_html, df, indicator, total_pages, new_vis = render_comments_explorer_view(
        comments_pool=comments_pool or [],
        total_video_comments=total_comments,
        sentiment_filter=sentiment_filter,
        search_query=search_query,
        sort_by=sort_by,
        page=new_page,
        page_size=page_size,
        visible_limit=visible_limit,
        use_load_more=False
    )
    return feed_html, df, indicator, new_page, new_vis


def on_comments_next_page(
    comments_pool: list,
    sentiment_filter: str,
    search_query: str,
    sort_by: str,
    page_size: str,
    current_page: int,
    visible_limit: int,
    total_comments: int = 0
):
    _, _, _, _, total_pages, _ = render_comments_explorer_view(
        comments_pool=comments_pool or [],
        total_video_comments=total_comments,
        sentiment_filter=sentiment_filter,
        search_query=search_query,
        sort_by=sort_by,
        page=current_page or 1,
        page_size=page_size,
        visible_limit=visible_limit,
        use_load_more=False
    )
    new_page = min(total_pages, (current_page or 1) + 1)
    _, feed_html, df, indicator, _, new_vis = render_comments_explorer_view(
        comments_pool=comments_pool or [],
        total_video_comments=total_comments,
        sentiment_filter=sentiment_filter,
        search_query=search_query,
        sort_by=sort_by,
        page=new_page,
        page_size=page_size,
        visible_limit=visible_limit,
        use_load_more=False
    )
    return feed_html, df, indicator, new_page, new_vis


def on_comments_load_more(
    comments_pool: list,
    sentiment_filter: str,
    search_query: str,
    sort_by: str,
    page_size: str,
    current_page: int,
    visible_limit: int,
    total_comments: int = 0
):
    step = 10 if str(page_size) == "10" else (25 if str(page_size) == "25" else 10)
    new_vis = (visible_limit or 10) + step
    _, feed_html, df, indicator, total_pages, final_vis = render_comments_explorer_view(
        comments_pool=comments_pool or [],
        total_video_comments=total_comments,
        sentiment_filter=sentiment_filter,
        search_query=search_query,
        sort_by=sort_by,
        page=1,
        page_size=page_size,
        visible_limit=new_vis,
        use_load_more=True
    )
    return feed_html, df, indicator, 1, final_vis


def run_youtube_intelligence_ui(
    url: str,
    api_key: str = "",
    comment_limit: str = "All Available Comments",
    comment_order: str = "Newest First (Time)",
    include_replies: bool = True
):
    empty_df = pd.DataFrame(columns=["#", "Author", "Sentiment", "Likes", "Published At", "Reply?", "Comment"])
    if not url or not url.strip():
        return (
            "⚠️ **Please enter a valid YouTube URL.**",
            "",
            "",
            "",
            None,
            None,
            None,
            {},
            [],
            0,
            "",
            "",
            empty_df,
            "Showing 0 comments",
            1,
            10
        )
    try:
        data = analyze_youtube_video(
            url.strip(),
            api_key=api_key.strip() if api_key else None,
            max_comments=comment_limit,
            comment_order=comment_order,
            include_replies=include_replies
        )
        m = data.get("metadata", {})
        t = data.get("transcript", {})
        s = data.get("sentiment", {})
        r = data.get("recommendations", {})
        comments_pool = s.get("classified_comments", [])
        total_on_yt = m.get("comment_count")

        views_str = f"{m.get('view_count'):,}" if m.get('view_count') is not None else "N/A (API Key Required)"
        likes_str = f"{m.get('like_count'):,}" if m.get('like_count') is not None else "N/A (API Key Required)"
        comments_str = f"{m.get('comment_count'):,}" if m.get('comment_count') is not None else "N/A (API Key Required)"

        meta_html = f"""
        <div style="background: #0f172a; border: 1px solid #334155; border-radius: 12px; padding: 18px; margin-bottom: 15px;">
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 280px; max-width: 380px;">
                    <img src="{m.get('thumbnail_url')}" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.5);" />
                </div>
                <div style="flex: 2; min-width: 300px;">
                    <span style="background: #2563eb; color: #fff; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: bold;">{m.get('source')}</span>
                    <h3 style="margin: 8px 0 6px 0; color: #f8fafc; font-size: 20px;">{m.get('title')}</h3>
                    <p style="margin: 0 0 12px 0; color: #94a3b8; font-size: 14px;"><strong>Channel:</strong> {m.get('channel')} | <strong>Duration:</strong> {m.get('duration')}</p>
                    <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px;">
                        <div style="background: #1e293b; padding: 8px 14px; border-radius: 8px; border: 1px solid #334155;">
                            <span style="color: #64748b; font-size: 12px;">Views</span>
                            <div style="color: #38bdf8; font-size: 16px; font-weight: bold;">{views_str}</div>
                        </div>
                        <div style="background: #1e293b; padding: 8px 14px; border-radius: 8px; border: 1px solid #334155;">
                            <span style="color: #64748b; font-size: 12px;">Likes</span>
                            <div style="color: #4ade80; font-size: 16px; font-weight: bold;">{likes_str}</div>
                        </div>
                        <div style="background: #1e293b; padding: 8px 14px; border-radius: 8px; border: 1px solid #334155;">
                            <span style="color: #64748b; font-size: 12px;">Comments</span>
                            <div style="color: #facc15; font-size: 16px; font-weight: bold;">{comments_str}</div>
                        </div>
                        <div style="background: #1e293b; padding: 8px 14px; border-radius: 8px; border: 1px solid #334155;">
                            <span style="color: #64748b; font-size: 12px;">Speaking Rate</span>
                            <div style="color: #c084fc; font-size: 16px; font-weight: bold;">{t.get('speaking_rate_wpm')} WPM</div>
                        </div>
                    </div>
                    <p style="margin: 0; color: #cbd5e1; font-size: 13px;"><em>{t.get('pace_category')}</em> • {t.get('word_count')} words</p>
                </div>
            </div>
            <div style="margin-top: 14px; padding: 8px 14px; background: #1e293b; border-left: 4px solid #f59e0b; border-radius: 4px; font-size: 12px; color: #fbbf24;">
                ℹ️ <strong>Authorized Analytics Notice:</strong> {m.get('authorized_analytics_note')}
            </div>
        </div>
        """

        pos_p = s.get("positive_percent", 0)
        neu_p = s.get("neutral_percent", 0)
        neg_p = s.get("negative_percent", 0)

        sentiment_html = f"""
        <div style="background: #0f172a; border: 1px solid #334155; border-radius: 12px; padding: 18px; margin-bottom: 15px;">
            <h4 style="margin: 0 0 10px 0; color: #f8fafc; font-size: 17px;">💬 Audience Sentiment: {s.get('sentiment_label')}</h4>
            <div style="background: #1e293b; border-radius: 8px; height: 18px; overflow: hidden; display: flex; margin-bottom: 14px;">
                <div style="background: #10b981; width: {pos_p}%;" title="Positive: {pos_p}%"></div>
                <div style="background: #64748b; width: {neu_p}%;" title="Neutral: {neu_p}%"></div>
                <div style="background: #ef4444; width: {neg_p}%;" title="Negative: {neg_p}%"></div>
            </div>
            <div style="display: flex; gap: 20px; font-size: 14px; margin-bottom: 16px;">
                <span style="color: #10b981; font-weight: bold;">🟢 Positive: {pos_p}%</span>
                <span style="color: #94a3b8; font-weight: bold;">⚪ Neutral: {neu_p}%</span>
                <span style="color: #ef4444; font-weight: bold;">🔴 Negative: {neg_p}%</span>
                <span style="color: #cbd5e1; margin-left: auto;">(Total Sampled: {s.get('total_analyzed')} comments)</span>
            </div>
            <div style="display: flex; gap: 16px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 250px; background: #1e293b; padding: 12px; border-radius: 8px; border-top: 3px solid #10b981;">
                    <strong style="color: #10b981;">🌟 Top Praised Elements:</strong>
                    <ul style="margin: 6px 0 0 16px; padding: 0; color: #e2e8f0; font-size: 13px;">
                        {''.join(f'<li>"{theme}"</li>' for theme in s.get('top_positive_themes', []))}
                    </ul>
                </div>
                <div style="flex: 1; min-width: 250px; background: #1e293b; padding: 12px; border-radius: 8px; border-top: 3px solid #ef4444;">
                    <strong style="color: #f87171;">⚠️ Constructive Feedback:</strong>
                    <ul style="margin: 6px 0 0 16px; padding: 0; color: #e2e8f0; font-size: 13px;">
                        {''.join(f'<li>"{crit}"</li>' for crit in s.get('constructive_criticism', []))}
                    </ul>
                </div>
            </div>
        </div>
        """

        tags_line = " ".join(f"`#{tag}`" for tag in r.get("seo_and_tags", []))
        recs_md = f"""### 🚀 AI-Powered Video Improvement Strategy (Score: **{r.get('overall_score')}/100**)

#### 🎯 Viewer Retention & Pacing
{chr(10).join(f"- {item}" for item in r.get('retention_recommendations', []))}

#### 🖼️ Title & Thumbnail Optimizations
{chr(10).join(f"- {item}" for item in r.get('title_and_thumbnail_improvements', []))}

#### 📚 Content Gaps & Follow-Up Concepts
{chr(10).join(f"- {item}" for item in r.get('content_gap_analysis', []))}

#### 🔍 SEO Keywords & Educational Tags
{tags_line}

> **💡 Executive Summary:**  
> {r.get('executive_summary')}
"""

        temp_dir = root_dir / "temp" / "youtube_reports"
        temp_dir.mkdir(parents=True, exist_ok=True)
        vid_id = data.get("video_id", "vid")

        md_exp = export_analysis_markdown(data, str(temp_dir / f"report_{vid_id}.md"))
        json_exp = export_analysis_json(data, str(temp_dir / f"data_{vid_id}.json"))
        csv_exp = export_analysis_csv(data, str(temp_dir / f"comments_{vid_id}.csv"))

        # Render Audience Comments Explorer
        summary_html, feed_html, df, page_indicator, total_pages, init_vis = render_comments_explorer_view(
            comments_pool=comments_pool,
            total_video_comments=total_on_yt,
            sentiment_filter="All Sentiments",
            search_query="",
            sort_by="Default (Retrieved)",
            page=1,
            page_size="10",
            visible_limit=10,
            use_load_more=False
        )

        status = f"✅ **Analysis Completed for:** {m.get('title')} (Retrieved {len(comments_pool)} comments)"
        return (
            status,
            meta_html,
            sentiment_html,
            recs_md,
            md_exp,
            json_exp,
            csv_exp,
            data,
            comments_pool,
            total_on_yt or 0,
            summary_html,
            feed_html,
            df,
            page_indicator,
            1,
            init_vis
        )

    except Exception as e:
        logger.error(f"Error in YouTube intelligence analysis: {e}\n{traceback.format_exc()}")
        return (
            f"❌ **Analysis Error**: {e}",
            "",
            "",
            "",
            None,
            None,
            None,
            {},
            [],
            0,
            "",
            f"""<div style="background: #450a0a; border: 1px solid #dc2626; border-radius: 8px; padding: 14px; color: #fca5a5;">❌ Analysis Failed: {html.escape(str(e))}</div>""",
            empty_df,
            "Error loading comments",
            1,
            10
        )


def chat_with_youtube_video_assistant(user_msg: str, chat_history: list, analysis_state: dict):
    if not user_msg or not user_msg.strip():
        return "", chat_history
    if not analysis_state or not analysis_state.get("metadata"):
        chat_history.append((user_msg, "⚠️ Please analyze a YouTube video first using the analyzer above to enable video intelligence chat!"))
        return "", chat_history

    m = analysis_state.get("metadata", {})
    t = analysis_state.get("transcript", {})
    s = analysis_state.get("sentiment", {})
    r = analysis_state.get("recommendations", {})

    context = f"""
    VIDEO CONTEXT:
    Title: {m.get('title')}
    Channel: {m.get('channel')}
    Duration: {m.get('duration')}
    Views: {m.get('view_count')}
    Speaking Rate: {t.get('speaking_rate_wpm')} WPM
    Transcript Summary: {t.get('summary_snippet')}
    Audience Sentiment: {s.get('sentiment_label')} ({s.get('positive_percent')}% positive)
    AI Recommendations: {r.get('retention_recommendations')}
    """

    prompt = f"Act as an expert YouTube Analyst for this specific video.\n{context}\n\nUser Question: {user_msg}\nProvide a concise, highly strategic, actionable answer."
    try:
        llm = get_llm()
        resp = llm.invoke([HumanMessage(content=prompt)])
        chat_history.append((user_msg, resp.content))
    except Exception as e:
        chat_history.append((user_msg, f"❌ Chat Error: {e}"))
    return "", chat_history




def generate_standalone_covers_ui(title: str, creator: str, trainer_img, logo_img, aspect_ratio: str):
    if not title or not title.strip():
        title = "Educational AI Masterclass"
    tr_path = trainer_img.name if hasattr(trainer_img, "name") else (str(trainer_img) if trainer_img else None)
    logo_path = logo_img.name if hasattr(logo_img, "name") else (str(logo_img) if logo_img else None)

    temp_dir = root_dir / "temp" / f"covers_{uuid.uuid4().hex[:6]}"
    covers = generate_vertical_video_covers(
        title=title.strip(),
        creator=creator.strip() if creator else "ai_educator_lab",
        trainer_img_path=tr_path,
        channel_logo_path=logo_path,
        output_dir=str(temp_dir),
        aspect_ratio=aspect_ratio
    )
    return covers, f"✅ Generated {len(covers)} {aspect_ratio} Video Covers successfully!"


def generate_standalone_notes_ui(title: str, script_text: str, creator: str):
    if not title or not title.strip():
        title = "Machine Learning Study Notes"
    temp_dir = root_dir / "temp" / f"notes_{uuid.uuid4().hex[:6]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    target_pdf = str(temp_dir / "handwritten_study_notes.pdf")

    pdf_path, png_paths = generate_handwritten_study_notes(
        video_title=title.strip(),
        transcript_text=script_text.strip() if script_text else "",
        output_pdf_path=target_pdf,
        creator=creator.strip() if creator else "ai_educator_lab"
    )
    return pdf_path, png_paths, f"✅ Compiled {len(png_paths)}-page Handwritten Study Notes PDF!"


def generate_standalone_engagement_ui(brand_name: str, logo_img):
    temp_dir = root_dir / "temp" / f"eng_{uuid.uuid4().hex[:6]}"
    assets_dict = generate_engagement_animation_assets(
        output_dir=str(temp_dir),
        brand_name=brand_name.strip() if brand_name else "ai_educator_lab"
    )
    return list(assets_dict.values()), "✅ Generated 5 Transparent Engagement Overlays (Like, Comment, Share, Follow, Bell)!"


# ==========================================
# Division B: 16-Stage Studio Editing UI Handler
# ==========================================

def run_automatic_studio_editing_ui(
    video_file,
    remove_pauses: bool,
    silence_thresh: float,
    target_lang: str,
    burn_subs: bool,
    add_edu: bool,
    add_eng: bool,
    gen_thumb: bool,
    gen_short: bool,
    gen_notes: bool,
    trainer_photo,
    channel_logo,
    progress=gr.Progress(track_tqdm=False)
):
    if not video_file:
        empty_card = render_stage_status_card(
            1, 16, "Input Validation", 0.0, 0.0, "failed",
            error_msg="Please upload a video file (.mp4, .mov, .mkv, .webm, .avi) to start automatic editing."
        )
        yield (
            None, None, None, None, None, [], [], None,
            empty_card,
            "⚠️ Upload required: No video file was provided.",
            gr.update(interactive=True),
            gr.update(interactive=False)
        )
        return

    vid_path = video_file.name if hasattr(video_file, "name") else str(video_file)
    trainer_p = trainer_photo.name if hasattr(trainer_photo, "name") else (str(trainer_photo) if trainer_photo else None)
    logo_p = channel_logo.name if hasattr(channel_logo, "name") else (str(channel_logo) if channel_logo else None)

    generator = run_automatic_video_editing_generator(
        video_path=vid_path,
        remove_pauses=remove_pauses,
        silence_threshold_db=silence_thresh,
        target_language=target_lang,
        burn_subtitles=burn_subs,
        add_educational_overlays=add_edu,
        add_engagement_animations=add_eng,
        generate_thumbnails=gen_thumb,
        generate_shorts=gen_short,
        generate_notes=gen_notes,
        trainer_img_path=trainer_p,
        channel_logo_path=logo_p
    )

    for state in generator:
        pct = max(0.0, min(1.0, state["progress_percent"] / 100.0))
        stage_name = state["stage_name"]
        stage_idx = state["stage_index"]
        progress(pct, desc=f"Stage {stage_idx}/16: {stage_name}")

        status_html = state["status_card_html"]
        logs = state["log_text"]

        if state["is_complete"]:
            outs = state.get("output_files", {})
            yield (
                outs.get("edited_video"),
                outs.get("vertical_reel") or outs.get("youtube_short"),
                outs.get("subtitles_srt"),
                outs.get("subtitles_vtt"),
                outs.get("edit_decision_list"),
                outs.get("thumbnails", []),
                outs.get("reel_covers", []),
                outs.get("study_notes_pdf"),
                status_html,
                logs,
                gr.update(interactive=True),
                gr.update(interactive=False)
            )
        elif state["is_error"] or state["is_cancelled"]:
            yield (
                None, None, None, None, None, [], [], None,
                status_html,
                logs,
                gr.update(interactive=True),
                gr.update(interactive=False)
            )
        else:
            yield (
                None, None, None, None, None, [], [], None,
                status_html,
                logs,
                gr.update(interactive=False),
                gr.update(interactive=True)
            )

def cancel_active_studio_job_ui():
    success, msg = cancel_active_studio_job()
    return f"🛑 {msg}", gr.update(interactive=False)


# ==========================================
# Gradio UI Construction (gr.Blocks)
# ==========================================

css = """
footer {visibility: hidden;}
.container {max-width: 1200px; margin: 0 auto;}
"""

with gr.Blocks(title="GenAI Video Intelligence Studio & Autonomous Production Platform") as demo:
    gr.Markdown("# 🎬 GenAI Video Intelligence Studio & Autonomous Production Platform")
    gr.Markdown("### 🚀 Enterprise 16-Stage Autonomous Video Studio, YouTube Intelligence, Multilingual Dubbing & Educational Content Engine (Ollama llama3.2:latest)")
    
    status_banner = gr.Markdown(value=get_ollama_status_markdown)
    
    with gr.Tabs():
        # ==============================================================
        # AUTONOMOUS VIDEO STUDIO & PRODUCTION PLATFORM
        # ==============================================================
        with gr.TabItem("🎬 Autonomous Video Studio"):
            with gr.Tabs():
                # SUB-TAB 1: 16-STAGE AUTONOMOUS VIDEO EDITOR
                with gr.TabItem("🎬 16-Stage Autonomous Video Editor"):
                    gr.Markdown("## 🎬 16-Stage Autonomous Educational Video Editing Studio")
                    gr.Markdown("Upload raw video. The multi-agent pipeline cuts awkward pauses, synchronizes multilingual captions, generates 1080x1920 vertical covers, creates vertical Shorts (<60s), and compiles handwritten-style study notes.")

                    with gr.Row():
                        with gr.Column(scale=3):
                            studio_video_input = gr.File(
                                label="🎥 Upload Raw Video (MP4, MOV, MKV, WEBM, AVI - up to 500MB)",
                                file_types=[".mp4", ".mov", ".mkv", ".webm", ".avi"]
                            )
                            with gr.Row():
                                studio_start_btn = gr.Button("🚀 Start 16-Stage Autonomous Editing", variant="primary", scale=2)
                                studio_cancel_btn = gr.Button("🛑 Cancel Active Job", variant="stop", scale=1, interactive=False)

                        with gr.Column(scale=2):
                            with gr.Accordion("⚙️ Advanced Studio Configuration", open=False):
                                studio_remove_pauses = gr.Checkbox(label="Intelligent Pause Removal", value=True)
                                studio_silence_thresh = gr.Slider(
                                    label="Silence Detection Threshold (dB)",
                                    minimum=-50.0,
                                    maximum=-15.0,
                                    value=-32.0,
                                    step=1.0
                                )
                                studio_target_lang = gr.Dropdown(
                                    label="Subtitle & Dubbing Target Language",
                                    choices=["English", "Kannada", "Telugu", "Hindi", "Tamil", "Malayalam", "Marathi", "Bengali", "Spanish", "French", "German", "Japanese"],
                                    value="English"
                                )
                                studio_burn_subs = gr.Checkbox(label="Burn Subtitles into Video", value=True)
                                studio_add_edu = gr.Checkbox(label="Context-Aware Educational Topic Badges", value=True)
                                studio_add_eng = gr.Checkbox(label="Engagement Overlays (Like/Comment/Share/Follow/Bell)", value=True)
                                studio_gen_thumbs = gr.Checkbox(label="Generate 3 Landscape Thumbnails (1280x720)", value=True)
                                studio_gen_shorts = gr.Checkbox(label="Generate YouTube Short / Vertical Video (<60s, Vertical 9:16)", value=True)
                                studio_gen_notes = gr.Checkbox(label="Generate Ruled Handwritten Study Notes (PDF & Images)", value=True)
                                studio_trainer_img = gr.File(label="Optional Trainer Photo (for Cover)", file_types=[".png", ".jpg", ".jpeg", ".webp"])
                                studio_logo_img = gr.File(label="Optional Channel Logo (for Cover)", file_types=[".png", ".jpg", ".jpeg", ".webp"])

                    studio_status_card = gr.HTML(
                        value=render_stage_status_card(
                            current_stage_idx=0,
                            total_stages=16,
                            stage_name="Ready for 16-Stage Autonomous Editing",
                            pct=0.0,
                            elapsed_sec=0.0,
                            status_label="queued",
                            detail_note="Upload an educational video above and click 'Start 16-Stage Autonomous Editing'."
                        )
                    )
                    studio_logs = gr.Textbox(label="Activity Diagnostics", lines=4, interactive=False)

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 🎥 Edited Master Video (MP4)")
                            studio_video_player = gr.Video(label="Final Edited Video")
                        with gr.Column():
                            gr.Markdown("### 📱 YouTube Short / Vertical Video (<60s, Vertical 9:16)")
                            studio_short_player = gr.Video(label="Vertical Video Candidate")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 🖼️ Auto-Generated Vertical Covers (1080×1920)")
                            studio_covers_gallery = gr.Gallery(label="Vertical Covers (3 Concepts)", columns=3, height=280)
                        with gr.Column():
                            gr.Markdown("### 🖼️ YouTube Thumbnails (1280×720)")
                            studio_gallery = gr.Gallery(label="Landscape Thumbnails", columns=3, height=280)

                    gr.Markdown("### 📥 Production Artifacts & Study Notes Downloads")
                    with gr.Row():
                        studio_srt_dl = gr.File(label="Subtitles (.srt)", interactive=False)
                        studio_vtt_dl = gr.File(label="Web Subtitles (.vtt)", interactive=False)
                        studio_edl_dl = gr.File(label="Edit Decision List (.json)", interactive=False)
                        studio_notes_dl = gr.File(label="Handwritten Notes (.pdf)", interactive=False)

                    studio_start_btn.click(
                        fn=run_automatic_studio_editing_ui,
                        inputs=[
                            studio_video_input,
                            studio_remove_pauses,
                            studio_silence_thresh,
                            studio_target_lang,
                            studio_burn_subs,
                            studio_add_edu,
                            studio_add_eng,
                            studio_gen_thumbs,
                            studio_gen_shorts,
                            studio_gen_notes,
                            studio_trainer_img,
                            studio_logo_img
                        ],
                        outputs=[
                            studio_video_player,
                            studio_short_player,
                            studio_srt_dl,
                            studio_vtt_dl,
                            studio_edl_dl,
                            studio_gallery,
                            studio_covers_gallery,
                            studio_notes_dl,
                            studio_status_card,
                            studio_logs,
                            studio_start_btn,
                            studio_cancel_btn
                        ]
                    )

                    studio_cancel_btn.click(
                        fn=cancel_active_studio_job_ui,
                        inputs=[],
                        outputs=[studio_logs, studio_cancel_btn]
                    )

                # SUB-TAB 2: VERTICAL COVER & THUMBNAIL GENERATOR (STANDALONE)
                with gr.TabItem("🖼️ Vertical Video Cover & Thumbnail Studio"):
                    gr.Markdown("## 🖼️ Vertical Video Cover & Thumbnail Studio")
                    gr.Markdown("Generate visually striking, high-contrast 1080×1920 vertical video covers complying with mobile safe margins (180px top, 220px bottom, 140px right).")

                    with gr.Row():
                        with gr.Column(scale=2):
                            cover_title_in = gr.Textbox(label="Video / Lesson Title", value="Complete Deep Learning in 60 Seconds", lines=2)
                            cover_creator_in = gr.Textbox(label="Creator Handle", value="ai_educator_lab")
                            cover_aspect_in = gr.Dropdown(label="Canvas Aspect Ratio", choices=["9:16 (Vertical Video)", "1:1 (Square)", "16:9 (Landscape)"], value="9:16 (Vertical Video)")
                            cover_trainer_in = gr.File(label="Optional Trainer Photo", file_types=[".png", ".jpg", ".jpeg", ".webp"])
                            cover_logo_in = gr.File(label="Optional Brand / Channel Logo", file_types=[".png", ".jpg", ".jpeg", ".webp"])
                            cover_gen_btn = gr.Button("Generate Video Covers 🎨", variant="primary")
                        with gr.Column(scale=3):
                            cover_status_out = gr.Markdown()
                            cover_gallery_out = gr.Gallery(label="Generated Video Covers (1080×1920)", columns=3, height=380)

                    cover_gen_btn.click(
                        fn=generate_standalone_covers_ui,
                        inputs=[cover_title_in, cover_creator_in, cover_trainer_in, cover_logo_in, cover_aspect_in],
                        outputs=[cover_gallery_out, cover_status_out]
                    )

                # SUB-TAB 3: HANDWRITTEN STUDY NOTES GENERATOR (STANDALONE)
                with gr.TabItem("📝 Handwritten Study Notes Generator"):
                    gr.Markdown("## 📝 Ruled Notebook Handwritten Study Notes Generator")
                    gr.Markdown("Convert any educational lecture or transcript into authentic cream-ruled A4 notebook pages complete with definitions, equations, diagrams, and multi-page PDF compilation.")

                    with gr.Row():
                        with gr.Column(scale=2):
                            notes_title_in = gr.Textbox(label="Lecture Title", value="Artificial Neural Networks & Backpropagation")
                            notes_creator_in = gr.Textbox(label="Creator / Institution Handle", value="ai_educator_lab")
                            notes_text_in = gr.Textbox(
                                label="Lecture Transcript / Content Text",
                                lines=10,
                                value="Neural networks are computational models inspired by the human brain. They consist of input, hidden, and output layers. The loss function calculates prediction error. Gradient descent updates weights using partial derivatives: W = W - alpha * dL/dW. Backpropagation applies the chain rule from calculus to propagate errors backwards."
                            )
                            notes_gen_btn = gr.Button("Compile Handwritten Notes 📄", variant="primary")
                        with gr.Column(scale=3):
                            notes_status_out = gr.Markdown()
                            notes_pdf_out = gr.File(label="📥 Download Handwritten Study Notes (.pdf)")
                            notes_gallery_out = gr.Gallery(label="Handwritten Notebook Pages", columns=2, height=380)

                    notes_gen_btn.click(
                        fn=generate_standalone_notes_ui,
                        inputs=[notes_title_in, notes_text_in, notes_creator_in],
                        outputs=[notes_pdf_out, notes_gallery_out, notes_status_out]
                    )

                # SUB-TAB 4: ENGAGEMENT ANIMATION STUDIO (STANDALONE)
                with gr.TabItem("🔔 Engagement Animation Studio"):
                    gr.Markdown("## 🔔 Like, Comment, Share, Follow & Notification Animations")
                    gr.Markdown("Generate transparent 2D/3D-style engagement overlays for educational and YouTube videos.")

                    with gr.Row():
                        with gr.Column(scale=2):
                            eng_brand_in = gr.Textbox(label="Brand / Creator Handle", value="ai_educator_lab")
                            eng_logo_in = gr.File(label="Optional Brand Logo", file_types=[".png", ".jpg", ".jpeg", ".webp"])
                            eng_gen_btn = gr.Button("Render Engagement Overlays 🔔", variant="primary")
                        with gr.Column(scale=3):
                            eng_status_out = gr.Markdown()
                            eng_gallery_out = gr.Gallery(label="Transparent Engagement Overlays", columns=3, height=350)

                    eng_gen_btn.click(
                        fn=generate_standalone_engagement_ui,
                        inputs=[eng_brand_in, eng_logo_in],
                        outputs=[eng_gallery_out, eng_status_out]
                    )

        # ==============================================================
        # PRESERVED: YOUTUBE VIDEO INTELLIGENCE & ANALYTICS
        # ==============================================================
        with gr.TabItem("📊 YouTube Intelligence & Analytics"):
            gr.Markdown("## 📊 YouTube Video Intelligence & Audience Analytics")
            gr.Markdown("Analyze any YouTube video, extract speech analytics, classify audience comment sentiment, and receive AI-powered retention & growth recommendations.")

            with gr.Row():
                yt_url_input = gr.Textbox(
                    label="YouTube Video URL",
                    placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ or https://youtu.be/...",
                    scale=4
                )
                yt_api_key_input = gr.Textbox(
                    label="YouTube API Key (Optional)",
                    placeholder="Leave blank to use configured API key or free public fallback",
                    type="password",
                    scale=2
                )
                yt_analyze_btn = gr.Button("Analyze Video 🚀", variant="primary", scale=1)

            # Comment Retrieval Window & Filtering Parameters
            with gr.Row():
                yt_comment_limit_input = gr.Dropdown(
                    label="🎯 Comment Filter / Retrieval Limit",
                    choices=["All Available Comments", "10", "25", "50", "100"],
                    value="All Available Comments",
                    info="Choose number of comments to fetch via YouTube API (10, 25, 50, 100, or All Available)",
                    scale=2
                )
                yt_comment_order_input = gr.Dropdown(
                    label="🔃 Comment Order",
                    choices=["Newest First (Time)", "Top / Most Relevant (Relevance)"],
                    value="Newest First (Time)",
                    info="Fetch newest chronological comments or top relevant comments",
                    scale=2
                )
                yt_include_replies_input = gr.Checkbox(
                    label="Include Thread Replies",
                    value=True,
                    info="Retrieve full replies in comment threads (ensuring all comments are captured)",
                    scale=1
                )

            yt_status = gr.Markdown()
            yt_meta_html = gr.HTML()
            yt_sentiment_html = gr.HTML()

            # Dedicated Audience Comments Explorer Section
            with gr.Accordion("💬 Audience Comments & Feedback Explorer (Full Interactive View)", open=True):
                yt_comments_summary_html = gr.HTML()

                with gr.Row():
                    yt_filter_sentiment = gr.Radio(
                        choices=["All Sentiments", "🟢 Positive", "⚪ Neutral", "🔴 Negative"],
                        value="All Sentiments",
                        label="Filter by Sentiment",
                        scale=3
                    )
                    yt_search_comments = gr.Textbox(
                        label="🔍 Search Comments",
                        placeholder="Search comments by keyword or author name...",
                        scale=3
                    )
                    yt_sort_comments = gr.Dropdown(
                        choices=["Default (Retrieved)", "Most Liked 👍", "Newest 🕒", "Oldest ⏳"],
                        value="Default (Retrieved)",
                        label="Sort Comments By",
                        scale=2
                    )
                    yt_page_size = gr.Dropdown(
                        choices=["10", "25", "50", "All"],
                        value="10",
                        label="Per Page",
                        scale=1
                    )

                with gr.Tabs():
                    with gr.TabItem("📋 Comment Cards"):
                        yt_comments_feed_html = gr.HTML()
                    with gr.TabItem("📊 Tabular Spreadsheet View"):
                        yt_comments_dataframe = gr.Dataframe(
                            headers=["#", "Author", "Sentiment", "Likes", "Published At", "Reply?", "Comment"],
                            datatype=["number", "str", "str", "number", "str", "str", "str"],
                            interactive=False,
                            wrap=True
                        )

                with gr.Row():
                    yt_prev_page_btn = gr.Button("◀ Previous Page", variant="secondary", scale=1)
                    yt_page_indicator = gr.Markdown("Showing 0 comments", scale=2)
                    yt_next_page_btn = gr.Button("Next Page ▶", variant="secondary", scale=1)
                    yt_load_more_btn = gr.Button("📥 Load More Comments (+10)", variant="primary", scale=2)
                    yt_refetch_comments_btn = gr.Button("🔄 Re-fetch / Expand from YouTube", variant="secondary", scale=2)

            yt_recs_md = gr.Markdown()

            with gr.Row():
                yt_md_download = gr.File(label="📥 Download Full Intelligence Report (.md)", interactive=False)
                yt_json_download = gr.File(label="📥 Download Structured Intelligence Data (.json)", interactive=False)
                yt_csv_download = gr.File(label="📥 Download Sentiment Comments (.csv)", interactive=False)

            gr.Markdown("### 🤖 Interactive Video Intelligence Assistant")
            gr.Markdown("Ask specific questions about this video's audience reaction, pacing, or retention strategy:")
            yt_chat_state = gr.State({})
            yt_comments_pool_state = gr.State([])
            yt_total_comments_state = gr.State(0)
            yt_current_page_state = gr.State(1)
            yt_visible_limit_state = gr.State(10)
            yt_chatbot = gr.Chatbot(label="Video Strategy Q&A", height=280)
            with gr.Row():
                yt_chat_msg = gr.Textbox(placeholder="Ask a question about the analyzed video (e.g., 'How can I fix the pacing?', 'Summarize viewer complaints')...", show_label=False, scale=4)
                yt_chat_btn = gr.Button("Ask Assistant 💡", variant="secondary", scale=1)

            # Analyze & Re-fetch Action Handlers
            for action_btn in [yt_analyze_btn, yt_refetch_comments_btn]:
                action_btn.click(
                    fn=run_youtube_intelligence_ui,
                    inputs=[
                        yt_url_input,
                        yt_api_key_input,
                        yt_comment_limit_input,
                        yt_comment_order_input,
                        yt_include_replies_input
                    ],
                    outputs=[
                        yt_status,
                        yt_meta_html,
                        yt_sentiment_html,
                        yt_recs_md,
                        yt_md_download,
                        yt_json_download,
                        yt_csv_download,
                        yt_chat_state,
                        yt_comments_pool_state,
                        yt_total_comments_state,
                        yt_comments_summary_html,
                        yt_comments_feed_html,
                        yt_comments_dataframe,
                        yt_page_indicator,
                        yt_current_page_state,
                        yt_visible_limit_state
                    ]
                )

            # Comment Filtering & Sorting Handlers
            for filter_trigger in [yt_filter_sentiment, yt_sort_comments, yt_page_size]:
                filter_trigger.change(
                    fn=on_comments_filter_change,
                    inputs=[
                        yt_comments_pool_state,
                        yt_filter_sentiment,
                        yt_search_comments,
                        yt_sort_comments,
                        yt_page_size,
                        yt_current_page_state,
                        yt_visible_limit_state,
                        yt_total_comments_state
                    ],
                    outputs=[
                        yt_comments_feed_html,
                        yt_comments_dataframe,
                        yt_page_indicator,
                        yt_current_page_state,
                        yt_visible_limit_state
                    ]
                )

            yt_search_comments.change(
                fn=on_comments_filter_change,
                inputs=[
                    yt_comments_pool_state,
                    yt_filter_sentiment,
                    yt_search_comments,
                    yt_sort_comments,
                    yt_page_size,
                    yt_current_page_state,
                    yt_visible_limit_state,
                    yt_total_comments_state
                ],
                outputs=[
                    yt_comments_feed_html,
                    yt_comments_dataframe,
                    yt_page_indicator,
                    yt_current_page_state,
                    yt_visible_limit_state
                ]
            )

            # Pagination Handlers
            yt_prev_page_btn.click(
                fn=on_comments_prev_page,
                inputs=[
                    yt_comments_pool_state,
                    yt_filter_sentiment,
                    yt_search_comments,
                    yt_sort_comments,
                    yt_page_size,
                    yt_current_page_state,
                    yt_visible_limit_state,
                    yt_total_comments_state
                ],
                outputs=[
                    yt_comments_feed_html,
                    yt_comments_dataframe,
                    yt_page_indicator,
                    yt_current_page_state,
                    yt_visible_limit_state
                ]
            )

            yt_next_page_btn.click(
                fn=on_comments_next_page,
                inputs=[
                    yt_comments_pool_state,
                    yt_filter_sentiment,
                    yt_search_comments,
                    yt_sort_comments,
                    yt_page_size,
                    yt_current_page_state,
                    yt_visible_limit_state,
                    yt_total_comments_state
                ],
                outputs=[
                    yt_comments_feed_html,
                    yt_comments_dataframe,
                    yt_page_indicator,
                    yt_current_page_state,
                    yt_visible_limit_state
                ]
            )

            yt_load_more_btn.click(
                fn=on_comments_load_more,
                inputs=[
                    yt_comments_pool_state,
                    yt_filter_sentiment,
                    yt_search_comments,
                    yt_sort_comments,
                    yt_page_size,
                    yt_current_page_state,
                    yt_visible_limit_state,
                    yt_total_comments_state
                ],
                outputs=[
                    yt_comments_feed_html,
                    yt_comments_dataframe,
                    yt_page_indicator,
                    yt_current_page_state,
                    yt_visible_limit_state
                ]
            )

            yt_chat_btn.click(
                fn=chat_with_youtube_video_assistant,
                inputs=[yt_chat_msg, yt_chatbot, yt_chat_state],
                outputs=[yt_chat_msg, yt_chatbot]
            )
            yt_chat_msg.submit(
                fn=chat_with_youtube_video_assistant,
                inputs=[yt_chat_msg, yt_chatbot, yt_chat_state],
                outputs=[yt_chat_msg, yt_chatbot]
            )

        # TAB 1: DASHBOARD
        with gr.TabItem("🏠 Dashboard"):
            gr.Markdown("## Welcome to your Personalized AI Learning Platform")
            gr.Markdown("Select any of the tabs above to launch specialized multi-agent tools!")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📚 Study Planner\nGenerate daily & weekly tailored plans.")
                with gr.Column():
                    gr.Markdown("### 📝 Exam Prep\nTake mock tests, quizzes, and important question sets.")
                with gr.Column():
                    gr.Markdown("### 💻 Coding Lab\nDebug code, optimize solutions, and learn DSA.")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🏗️ Project Architect\nGenerate step-by-step guides and database schemas.")
                with gr.Column():
                    gr.Markdown("### 📄 Document Analyzer\nSummarize and query large documents.")
                with gr.Column():
                    gr.Markdown("### 🎬 3D Cartoon Studio\nGenerate animated explainer videos in 30+ languages.")
                    
        # TAB 2: AI CHAT ASSISTANT
        with gr.TabItem("💬 AI Chat Assistant"):
            gr.Markdown("## 💬 Persistent AI Multi-Modal Chatbot & Media Studio")
            
            with gr.Row():
                session_selector = gr.Dropdown(
                    label="Previous Conversations",
                    choices=get_session_choices(),
                    value=get_default_session_id(),
                    interactive=True,
                    scale=3
                )
                new_chat_btn = gr.Button("➕ New Chat", variant="secondary", scale=1)
                delete_chat_btn = gr.Button("🗑️ Delete Selected Chat", variant="stop", scale=1)
                delete_all_btn = gr.Button("⚠️ Delete All History", variant="stop", scale=1)

            # Mode Selector Radio Bar
            chatbot_mode = gr.Radio(
                choices=["💬 Chat", "🖼️ Text → Image", "🖼️ Image → Image", "🎬 Text → Video", "🎬 Image → Video"],
                value="💬 Chat",
                label="🎯 Select Chatbot Generation Mode",
                interactive=True
            )
            
            gen_ref_image = gr.Image(
                label="📷 Upload Reference Image for Image → Image / Image → Video",
                type="filepath",
                visible=False
            )

            chatbot = gr.Chatbot(label="Conversation History", height=450, value=get_initial_messages)
            current_session_id = gr.State(get_default_session_id)
            
            with gr.Accordion("📎 Attach Files & Images to Chat", open=False):
                with gr.Row():
                    file_upload = gr.File(
                        label="📄 Upload Documents (PDF, DOCX, PPTX, TXT, Code, CSV)",
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".pptx", ".txt", ".py", ".cpp", ".java", ".json", ".csv", ".md"],
                        scale=1
                    )
                    image_upload = gr.File(
                        label="🖼️ Upload Images (PNG, JPG, JPEG, WEBP, BMP)",
                        file_count="multiple",
                        file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
                        scale=1
                    )

            with gr.Row():
                msg_input = gr.Textbox(placeholder="Ask me anything or attach files/images...", show_label=False, scale=4)
                send_btn = gr.Button("Send 🚀", variant="primary", scale=1)
                
            download_output = gr.File(label="📥 Generated Media Download (Click to save directly to local device)", interactive=False)

            chatbot_mode.change(
                fn=update_mode_ui,
                inputs=[chatbot_mode],
                outputs=[gen_ref_image, msg_input]
            )

            send_btn.click(
                fn=chat_respond,
                inputs=[msg_input, chatbot, current_session_id, chatbot_mode, gen_ref_image, file_upload, image_upload],
                outputs=[msg_input, chatbot, current_session_id, session_selector, gen_ref_image, file_upload, image_upload, download_output]
            )
            
            msg_input.submit(
                fn=chat_respond,
                inputs=[msg_input, chatbot, current_session_id, chatbot_mode, gen_ref_image, file_upload, image_upload],
                outputs=[msg_input, chatbot, current_session_id, session_selector, gen_ref_image, file_upload, image_upload, download_output]
            )
            
            session_selector.change(
                fn=select_session,
                inputs=[session_selector],
                outputs=[chatbot, current_session_id]
            )
            
            new_chat_btn.click(
                fn=start_new_chat,
                inputs=[],
                outputs=[chatbot, current_session_id, session_selector, gen_ref_image, file_upload, image_upload, download_output]
            )
            
            delete_chat_btn.click(
                fn=delete_selected_chat,
                inputs=[current_session_id],
                outputs=[chatbot, current_session_id, session_selector, gen_ref_image, file_upload, image_upload, download_output]
            )
            
            delete_all_btn.click(
                fn=delete_all_chat_history,
                inputs=[],
                outputs=[chatbot, current_session_id, session_selector, gen_ref_image, file_upload, image_upload, download_output]
            )

        # TAB 3: STUDY PLANNER
        with gr.TabItem("📅 Study Planner"):
            gr.Markdown("## 📅 AI Study Planner")
            gr.Markdown("Create a personalized, AI-generated study schedule tailored to your goals and time constraints.")
            
            with gr.Row():
                with gr.Column():
                    sp_topic = gr.Textbox(label="Subject or Topic", placeholder="e.g., Python Programming, Calculus")
                    sp_level = gr.Dropdown(label="Current Knowledge Level", choices=["Beginner", "Intermediate", "Advanced"], value="Beginner")
                    sp_duration = gr.Dropdown(label="Plan Duration", choices=["1 Day", "3 Days", "1 Week", "2 Weeks", "1 Month"], value="1 Week")
                with gr.Column():
                    sp_hours = gr.Slider(label="Hours available per day", minimum=1, maximum=12, value=2, step=1)
                    sp_goals = gr.Textbox(label="Specific Goals or Focus Areas", placeholder="e.g., Focus on mastering loops and functions", lines=3)
                    
            sp_submit = gr.Button("Generate Study Plan 🧠", variant="primary")
            sp_output = gr.Markdown(label="Generated Plan")
            sp_download = gr.File(label="📥 Download Study Plan (.md)")
            
            sp_submit.click(
                fn=generate_study_plan,
                inputs=[sp_topic, sp_level, sp_duration, sp_hours, sp_goals],
                outputs=[sp_output, sp_download]
            )

        # TAB 4: EXAM PREPARATION
        with gr.TabItem("📝 Exam Preparation"):
            gr.Markdown("## 📝 AI Exam Preparation & Mock Tests")
            
            with gr.Row():
                with gr.Column():
                    ep_topic = gr.Textbox(label="Subject or Topic", placeholder="e.g., Biology, React.js, World History")
                    ep_type = gr.Dropdown(label="What do you want to generate?", choices=["Mock Test (Multiple Choice)", "Important Questions", "Short Quiz", "Flashcard Concepts"], value="Mock Test (Multiple Choice)")
                with gr.Column():
                    ep_diff = gr.Dropdown(label="Difficulty Level", choices=["Beginner", "Intermediate", "Advanced", "University Level"], value="Intermediate")
                    ep_num = gr.Number(label="Number of Questions/Topics", value=10, precision=0)
                    
            ep_context = gr.Textbox(label="Any specific focus areas or syllabus details?", placeholder="e.g., Focus on Chapter 3", lines=2)
            ep_submit = gr.Button("Generate Exam Material 📝", variant="primary")
            ep_output = gr.Markdown(label="Generated Material")
            ep_download = gr.File(label="📥 Download Exam Material (.md)")
            
            ep_submit.click(
                fn=generate_exam_material,
                inputs=[ep_topic, ep_type, ep_diff, ep_num, ep_context],
                outputs=[ep_output, ep_download]
            )

        # TAB 5: CODING LAB
        with gr.TabItem("💻 Coding Lab"):
            gr.Markdown("## 💻 AI Coding Lab & Mentor")
            
            cl_mode = gr.Radio(label="What do you want to do?", choices=["Debug Code", "Explain Code", "Optimize Code", "Learn a Topic (e.g., DSA)"], value="Debug Code")
            cl_lang = gr.Dropdown(label="Programming Language", choices=["Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "Other"], value="Python")
            cl_code = gr.Textbox(label="Paste your code here:", lines=6, placeholder="def hello_world():\n    print('Hello')")
            cl_error = gr.Textbox(label="Error Message (Optional):", placeholder="Paste the error traceback")
            cl_dsa = gr.Textbox(label="Topic to Learn (if DSA mode selected):", placeholder="e.g., Binary Search Trees")
            
            cl_submit = gr.Button("Run Analysis 👨‍💻", variant="primary")
            cl_output = gr.Markdown(label="Analysis Output")
            cl_download = gr.File(label="📥 Download Analysis (.md)")
            
            cl_submit.click(
                fn=run_coding_lab,
                inputs=[cl_mode, cl_lang, cl_code, cl_error, cl_dsa],
                outputs=[cl_output, cl_download]
            )

        # TAB 6: PROJECT BUILDER
        with gr.TabItem("🏗️ Project Builder"):
            gr.Markdown("## 🏗️ AI Project Builder & Architect")
            
            pb_idea = gr.Textbox(label="Project Idea / Description", placeholder="e.g., A real-time chat app, E-commerce platform")
            pb_stack = gr.Textbox(label="Preferred Tech Stack (Optional)", placeholder="e.g., React, Node.js, MongoDB")
            
            with gr.Row():
                pb_type = gr.Dropdown(label="What do you want to generate?", choices=[
                    "Complete Step-by-Step Guide",
                    "System Architecture Design",
                    "Module / Component Breakdown",
                    "Database Schema",
                    "README Documentation"
                ], value="Complete Step-by-Step Guide")
                pb_exp = gr.Dropdown(label="Your Experience Level", choices=["Beginner", "Intermediate", "Advanced"], value="Intermediate")
                
            pb_submit = gr.Button("Generate Project Plan 🏗️", variant="primary")
            pb_output = gr.Markdown(label="Project Architecture")
            pb_download = gr.File(label="📥 Download Project Plan (.md)")
            
            pb_submit.click(
                fn=generate_project_plan,
                inputs=[pb_idea, pb_stack, pb_type, pb_exp],
                outputs=[pb_output, pb_download]
            )

        # TAB 7: DOCUMENT INTELLIGENCE & REAL-TIME PROGRESS MONITOR
        with gr.TabItem("📄 Document Intelligence & Monitor"):
            gr.Markdown("## 📄 Enterprise Document Intelligence & Real-Time Progress Monitor")
            gr.Markdown("Upload, edit, chunk, generate dense embeddings, and analyze enterprise documents across 8 real-time stages with persistent progress tracking, versioning, and cancellation.")

            with gr.Row():
                with gr.Column(scale=3):
                    doc_file = gr.File(label="Upload Document (PDF, DOCX, PPTX, TXT)")
                    doc_text = gr.Textbox(
                        label="Document Content / Direct Text Editor:",
                        lines=7,
                        placeholder="Paste document text or edit existing content for versioned re-indexing..."
                    )
                with gr.Column(scale=2):
                    doc_task = gr.Dropdown(
                        label="Intelligence & Processing Operation",
                        choices=[
                            "Comprehensive Document Intelligence",
                            "Summarize Document",
                            "Extract Key Takeaways & Action Items",
                            "Extract Structured Data & Tables",
                            "Compare with Another Document",
                            "Ask a Specific Question (RAG Query)",
                            "Edit & Reindex Existing Document (Versioned Update)"
                        ],
                        value="Comprehensive Document Intelligence"
                    )
                    doc_q = gr.Textbox(label="Specific Query / Prompt (for RAG Questioning):", placeholder="Enter specific query or focus area...")
                    doc_compare_file = gr.File(label="Optional Second Document (for Comparison)", file_types=[".pdf", ".docx", ".pptx", ".txt"])
                    doc_id_input = gr.Textbox(label="Optional Existing Document ID (for Versioned Update)", placeholder="Leave blank to create new document ID")

            with gr.Row():
                doc_submit = gr.Button("🚀 Process & Index Document", variant="primary", scale=2)
                doc_cancel_btn = gr.Button("🛑 Cancel Operation", variant="stop", scale=1, interactive=False)
                doc_refresh_queue_btn = gr.Button("🔄 Refresh Queue", variant="secondary", scale=1)

            doc_job_id_state = gr.State("")

            doc_status_card = gr.HTML(
                value=render_document_status_card(
                    current_stage_idx=0,
                    total_stages=8,
                    stage_name="Ready for Document Processing",
                    pct=0.0,
                    elapsed_sec=0.0,
                    status_label="pending",
                    filename="Select or upload document above",
                    detail_note="Choose an operation and click 'Process & Index Document' to track the 8-stage workflow."
                )
            )

            doc_logs = gr.Textbox(label="Processing Activity Diagnostics", lines=3, interactive=False)

            with gr.Row():
                with gr.Column(scale=3):
                    doc_output = gr.Markdown(label="Intelligence Analysis Results")
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Document Exports")
                    doc_download = gr.File(label="Analysis Report (.md)", interactive=False)
                    doc_download_json = gr.File(label="Search Index & Chunks (.json)", interactive=False)

            gr.Markdown("### 📋 Active Document Processing Queue & Version History")
            doc_queue_table = gr.DataFrame(
                value=get_document_queue_dataframe,
                label="Document Processing Queue",
                wrap=True
            )

            doc_submit.click(
                fn=run_document_intelligence_ui,
                inputs=[doc_file, doc_text, doc_task, doc_q, doc_compare_file, doc_id_input],
                outputs=[
                    doc_status_card,
                    doc_output,
                    doc_download,
                    doc_download_json,
                    doc_queue_table,
                    doc_logs,
                    doc_job_id_state,
                    doc_submit,
                    doc_cancel_btn
                ]
            )

            doc_cancel_btn.click(
                fn=cancel_active_document_job_ui,
                inputs=[doc_job_id_state],
                outputs=[doc_logs, doc_cancel_btn]
            )

            doc_refresh_queue_btn.click(
                fn=get_document_queue_dataframe,
                inputs=[],
                outputs=[doc_queue_table]
            )

        # TAB 8: AUDIO STUDIO
        with gr.TabItem("🎧 Audio Learning"):
            gr.Markdown("## 🎧 Audio Learning System")
            gr.Markdown("Convert any text, summary, or note into spoken audio using Edge-TTS.")
            
            aud_text = gr.Textbox(label="Enter text to convert to audio:", lines=6, placeholder="Enter text here...")
            aud_submit = gr.Button("Generate Audio 🎵", variant="primary")
            aud_status = gr.Markdown()
            aud_player = gr.Audio(label="Audio Player", type="filepath")
            aud_file = gr.File(label="Download Audio MP3")
            
            aud_submit.click(
                fn=run_audio_generation,
                inputs=[aud_text],
                outputs=[aud_player, aud_file, aud_status]
            )

        # TAB 9: 3D CARTOON VIDEO STUDIO
        with gr.TabItem("🎬 3D Video Studio"):
            gr.Markdown("## 🎬 3D Multilingual Educational Cartoon Studio")
            gr.Markdown("### Featuring **Professor RoboCat** — Friendly Blue Robotic Cat Teacher 🤖🐾")
            
            with gr.Row():
                with gr.Column():
                    vid_mode = gr.Radio(label="Input Source", choices=["Text / Script", "Upload Document (PDF/DOCX/PPTX/TXT)"], value="Text / Script")
                    vid_script = gr.Textbox(label="Educational Script", lines=5, value="Artificial Neural Networks are computing systems inspired by human brains, using layers of nodes to process data and learn patterns.")
                    vid_doc = gr.File(label="Upload Lesson File")
                    
                    vid_lang = gr.Dropdown(label="Narration Language", choices=[
                        "Auto-Detect", "English", "Hindi", "Telugu", "Kannada", "Tamil", "Malayalam",
                        "Marathi", "Gujarati", "Bengali", "Punjabi", "Urdu", "French", "German",
                        "Spanish", "Portuguese", "Italian", "Japanese", "Chinese", "Korean",
                        "Arabic", "Russian", "Turkish", "Dutch", "Thai", "Vietnamese", "Indonesian", "Malay"
                    ], value="English")
                    
                with gr.Column():
                    vid_persona = gr.Dropdown(label="Voice Persona", choices=["Teacher Female", "Teacher Male", "Cartoon Child"], value="Teacher Female")
                    vid_theme = gr.Dropdown(label="Environment Backdrop", choices=["Science Lab", "Space Observatory", "AI Neural Laboratory", "Digital Classroom", "Nature Observatory"], value="AI Neural Laboratory")
                    vid_prop = gr.Dropdown(label="3D Interactive Topic Prop", choices=["AI Neural Networks", "Coding & Flowcharts", "Science Atom & Physics"], value="AI Neural Networks")
                    vid_res = gr.Dropdown(label="Quality Resolution", choices=["1080p (Full HD)", "720p (HD)", "4K (Ultra HD)"], value="1080p (Full HD)")
                    vid_aspect = gr.Dropdown(label="Aspect Ratio", choices=["16:9 (Widescreen)", "9:16 (Vertical Reel/Short)", "1:1 (Square)"], value="16:9 (Widescreen)")
                    
            with gr.Row():
                sb_btn = gr.Button("📋 Preview Storyboard Breakdown")
                vid_submit = gr.Button("🚀 Render Multilingual 3D Cartoon Video", variant="primary")
                
            sb_output = gr.Markdown()
            vid_status = gr.Markdown()
            
            with gr.Row():
                vid_player = gr.Video(label="Rendered MP4 Video")
                vid_audio = gr.Audio(label="Narration Speech MP3", type="filepath")
                vid_srt = gr.File(label="Subtitles SRT File")
                
            sb_btn.click(
                fn=preview_storyboard,
                inputs=[vid_script],
                outputs=[sb_output]
            )
            
            vid_submit.click(
                fn=run_video_generation,
                inputs=[vid_mode, vid_script, vid_doc, vid_lang, vid_persona, vid_theme, vid_prop, vid_res, vid_aspect],
                outputs=[vid_player, vid_audio, vid_srt, vid_status]
            )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, allowed_paths=[str(root_dir)])
