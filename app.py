import streamlit as st
from openai import OpenAI
import uuid
import json
from datetime import datetime
from typing import List, Dict, Optional

# -----------------------------
# Session Initialization
# -----------------------------
def init_session_state():
    if "tasks" not in st.session_state:
        st.session_state.tasks = []
    if "proposed_subtasks" not in st.session_state:
        st.session_state.proposed_subtasks = None
    if "last_plan" not in st.session_state:
        st.session_state.last_plan = ""
    if "filter_query" not in st.session_state:
        st.session_state.filter_query = ""


# -----------------------------
# Data Model Helpers
# -----------------------------
PRIORITIES = ["Low", "Medium", "High"]

def new_task(title: str, due: Optional[str], priority: str, notes: str = "", completed: bool = False) -> Dict:
    return {
        "id": uuid.uuid4().hex,
        "title": title.strip(),
        "due": due.strip() if due else "",
        "priority": priority,
        "notes": notes.strip(),
        "completed": completed,
        "created_at": datetime.utcnow().isoformat()
    }

def priority_index(priority: str) -> int:
    try:
        return PRIORITIES.index(priority)
    except ValueError:
        return 1


# -----------------------------
# OpenAI Client and AI Utilities
# -----------------------------
def get_client() -> OpenAI:
    # Requires OPENAI_API_KEY to be set in environment or Streamlit secrets.
    # The OpenAI SDK will automatically pick it up.
    return OpenAI()

def call_chat_completion(messages: List[Dict[str, str]], model: str = "gpt-4", temperature: float = 0.2) -> str:
    client = get_client()
    try:
        response = client.chat.completions.create(
            model=model,  # "gpt-4" or "gpt-3.5-turbo"
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def ai_generate_plan(tasks: List[Dict], model: str, temperature: float, extra_context: str = "") -> str:
    if not tasks:
        return "No tasks to plan. Add some tasks first."
    tasks_summary = []
    for t in tasks:
        status = "Done" if t["completed"] else "Pending"
        tasks_summary.append(
            f"- {t['title']} [Priority: {t['priority']} | Due: {t['due'] or 'N/A'} | Status: {status}]"
        )
    prompt = (
        "You are a helpful assistant that organizes to-do lists and creates a prioritized, actionable plan for the day. "
        "Consider due dates, priorities, and realistic time estimates. Present a concise ordered checklist with time estimates and brief notes.\n\n"
        "User tasks:\n" + "\n".join(tasks_summary) + "\n\n"
        f"Additional context/preferences (optional): {extra_context}\n\n"
        "Output format:\n"
        "1) A brief intro line.\n"
        "2) An ordered list of tasks with estimated time (e.g., 30m, 1h), and short rationale.\n"
        "3) If helpful, suggest small breaks.\n"
        "4) A short wrap-up tip."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    return call_chat_completion(messages, model=model, temperature=temperature)

def ai_breakdown_task(task_title: str, task_context: str, model: str, temperature: float) -> str:
    prompt = (
        f"Break down the task into clear, small, actionable subtasks:\n\n"
        f"Task: {task_title}\n"
        f"Context/Notes: {task_context or 'None'}\n\n"
        "Guidelines:\n"
        "- Use concise imperative phrasing.\n"
        "- 5–10 steps if possible.\n"
        "- Avoid redundancy.\n"
        "Return as a simple numbered list. No extra commentary."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    return call_chat_completion(messages, model=model, temperature=temperature)

def parse_subtasks(text: str) -> List[str]:
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove common list markers
        for prefix in ["- ", "* ", "• "]:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        # Remove numeric prefixes like "1. " or "1) "
        if len(line) > 2 and (line[1] == "." or line[1] == ")") and line[0].isdigit():
            line = line[2:].strip()
        # Remove multi-digit numeric prefixes
        while len(line) > 3 and line[0].isdigit():
            # Find number prefix until first non-digit
            j = 0
            while j < len(line) and line[j].isdigit():
                j += 1
            if j < len(line) and (line[j] in [".", ")"]):
                line = line[j+1:].strip()
            else:
                break
        if line:
            items.append(line)
    # Deduplicate while preserving order
    seen = set()
    unique_items = []
    for i in items:
        if i not in seen:
            seen.add(i)
            unique_items.append(i)
    return unique_items


# -----------------------------
# UI Helpers
# -----------------------------
def add_task_form():
    st.subheader("Add a new task")
    with st.form("add_task_form", clear_on_submit=True):
        title = st.text_input("Task title", placeholder="e.g., Prepare project proposal")
        col1, col2 = st.columns([1, 1])
        with col1:
            due = st.text_input("Due (optional, free text)", placeholder="e.g., 2025-08-20 or 'Friday'")
        with col2:
            priority = st.selectbox("Priority", PRIORITIES, index=1)
        notes = st.text_area("Notes (optional)", placeholder="Details, links, acceptance criteria...")
        submitted = st.form_submit_button("Add Task")
        if submitted:
            if not title.strip():
                st.warning("Please enter a task title.")
            else:
                st.session_state.tasks.append(new_task(title, due, priority, notes))
                st.success("Task added.")

def render_task_item(task: Dict, idx: int):
    cid = task["id"]
    cols = st.columns([0.1, 0.45, 0.2, 0.15, 0.1])
    with cols[0]:
        completed = st.checkbox("", value=task["completed"], key=f"completed_{cid}", help="Mark complete")
    with cols[1]:
        title = st.text_input("Title", value=task["title"], key=f"title_{cid}", label_visibility="collapsed")
    with cols[2]:
        due = st.text_input("Due", value=task["due"], key=f"due_{cid}", label_visibility="collapsed")
    with cols[3]:
        priority = st.selectbox("Priority", PRIORITIES, index=priority_index(task["priority"]), key=f"priority_{cid}", label_visibility="collapsed")
    with cols[4]:
        delete_clicked = st.button("🗑️", key=f"delete_{cid}", help="Delete task")

    # Notes in an expander
    with st.expander("Notes", expanded=False):
        notes = st.text_area("Details", value=task["notes"], key=f"notes_{cid}", height=80)

    # Update the task object
    updated = {
        "id": cid,
        "title": title.strip(),
        "due": due.strip(),
        "priority": priority,
        "notes": notes.strip(),
        "completed": completed,
        "created_at": task.get("created_at", datetime.utcnow().isoformat()),
    }
    st.session_state.tasks[idx] = updated
    return delete_clicked

def tasks_list(filter_query: str = "", show_completed: bool = True):
    st.subheader("Your tasks")
    tasks = st.session_state.tasks

    # Apply filters
    filtered = []
    q = filter_query.lower().strip()
    for t in tasks:
        if not show_completed and t["completed"]:
            continue
        if q:
            blob = f"{t['title']} {t['due']} {t['priority']} {t['notes']}".lower()
            if q not in blob:
                continue
        filtered.append(t)

    if not filtered:
        st.info("No tasks match your filter." if (q or not show_completed) else "No tasks yet. Add some above!")
        return

    # Sort by completion, then priority, then due presence
    filtered = sorted(filtered, key=lambda x: (x["completed"], -priority_index(x["priority"]), x["due"] or "zzzz"), reverse=False)
    to_delete = []
    for idx, t in enumerate(filtered):
        st.markdown("---")
        if render_task_item(t, st.session_state.tasks.index(t)):
            to_delete.append(t["id"])

    # Delete after rendering to avoid index shifts
    if to_delete:
        st.session_state.tasks = [t for t in st.session_state.tasks if t["id"] not in to_delete]
        st.success(f"Deleted {len(to_delete)} task(s).")


def export_import_controls():
    st.subheader("Import/Export")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export tasks as JSON"):
            data = json.dumps(st.session_state.tasks, indent=2)
            st.download_button("Download tasks.json", data=data, file_name="tasks.json", mime="application/json")
    with col2:
        uploaded = st.file_uploader("Import JSON", type=["json"], accept_multiple_files=False)
        if uploaded is not None:
            try:
                data = json.load(uploaded)
                if isinstance(data, list):
                    # Basic validation for required keys
                    cleaned = []
                    for item in data:
                        if isinstance(item, dict) and "title" in item:
                            # Ensure required keys exist
                            cleaned.append(new_task(
                                title=item.get("title", ""),
                                due=item.get("due", ""),
                                priority=item.get("priority", "Medium") if item.get("priority") in PRIORITIES else "Medium",
                                notes=item.get("notes", ""),
                                completed=bool(item.get("completed", False))
                            ))
                    st.session_state.tasks.extend(cleaned)
                    st.success(f"Imported {len(cleaned)} tasks.")
                else:
                    st.error("Invalid JSON format. Expected a list of tasks.")
            except Exception as e:
                st.error(f"Failed to import: {e}")


# -----------------------------
# Main Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="AI To-Do List Agent", page_icon="✅", layout="wide")
    init_session_state()

    # Sidebar Settings
    st.sidebar.title("Settings")
    model = st.sidebar.selectbox("OpenAI Model", options=["gpt-4", "gpt-3.5-turbo"], index=0)
    temperature = st.sidebar.slider("Creativity (temperature)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    st.sidebar.markdown("---")
    extra_context = st.sidebar.text_area("Planning preferences (optional)", placeholder="e.g., Focus mornings on deep work; limit to 6h total.")
    st.sidebar.markdown("---")
    st.sidebar.caption("Note: Set OPENAI_API_KEY in your environment to enable AI features.")

    st.title("Task To-Do List Agent")
    add_task_form()

    # Filters
    st.markdown("---")
    colf1, colf2, colf3 = st.columns([0.5, 0.3, 0.2])
    with colf1:
        st.session_state.filter_query = st.text_input("Filter tasks", value=st.session_state.filter_query, placeholder="Search by text, priority, due, notes")
    with colf2:
        show_completed = st.checkbox("Show completed", value=True)
    with colf3:
        if st.button("Clear completed"):
            before = len(st.session_state.tasks)
            st.session_state.tasks = [t for t in st.session_state.tasks if not t["completed"]]
            after = len(st.session_state.tasks)
            st.success(f"Cleared {before - after} completed task(s).")

    tasks_list(filter_query=st.session_state.filter_query, show_completed=show_completed)

    # AI Tools
    st.markdown("### AI Assistant")
    aic1, aic2 = st.columns([0.5, 0.5])
    with aic1:
        if st.button("Generate prioritized plan"):
            with st.spinner("Thinking..."):
                plan = ai_generate_plan(st.session_state.tasks, model=model, temperature=temperature, extra_context=extra_context)
                st.session_state.last_plan = plan
    with aic2:
        task_options = [(t["title"], t["id"]) for t in st.session_state.tasks if not t["completed"]]
        selected_task_id = st.selectbox("Break down a task", options=[tid for _, tid in task_options], format_func=lambda tid: next((title for title, id_ in task_options if id_ == tid), "Select a task")) if task_options else None
        if st.button("Break down") and selected_task_id:
            parent = next((t for t in st.session_state.tasks if t["id"] == selected_task_id), None)
            if parent:
                with st.spinner("Decomposing task..."):
                    result = ai_breakdown_task(parent["title"], parent.get("notes", ""), model=model, temperature=temperature)
                    items = parse_subtasks(result)
                    st.session_state.proposed_subtasks = {
                        "parent_id": parent["id"],
                        "parent_title": parent["title"],
                        "items": items
                    }

    if st.session_state.last_plan:
        st.markdown("#### Suggested Plan")
        st.write(st.session_state.last_plan)

    if st.session_state.proposed_subtasks:
        st.markdown("#### Proposed Subtasks")
        parent_title = st.session_state.proposed_subtasks["parent_title"]
        st.caption(f"For: {parent_title}")
        selected_flags = []
        for i, item in enumerate(st.session_state.proposed_subtasks["items"]):
            selected = st.checkbox(item, value=True, key=f"subtask_select_{i}")
            selected_flags.append(selected)
        col_add1, col_add2 = st.columns([0.2, 0.8])
        with col_add1:
            if st.button("Add selected"):
                parent = next((t for t in st.session_state.tasks if t["id"] == st.session_state.proposed_subtasks["parent_id"]), None)
                if parent:
                    count = 0
                    for include, title in zip(selected_flags, st.session_state.proposed_subtasks["items"]):
                        if include:
                            st.session_state.tasks.append(
                                new_task(
                                    title=title,
                                    due=parent.get("due", ""),
                                    priority=parent.get("priority", "Medium"),
                                    notes=f"Subtask of: {parent_title}"
                                )
                            )
                            count += 1
                    st.success(f"Added {count} subtasks.")
                st.session_state.proposed_subtasks = None
        with col_add2:
            if st.button("Dismiss subtasks"):
                st.session_state.proposed_subtasks = None

    st.markdown("---")
    export_import_controls()


if __name__ == "__main__":
    main()