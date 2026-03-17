"""
Omaha System Mapper — Interactive Evaluation & Prompt Refinement

Run locally:
    cd omaha_mapping
    streamlit run streamlit_app.py

Workflow:
    1. Enter your OpenAI API key in the sidebar (stored in session only)
    2. Upload the Omaha Excel file (once — embedding index is cached)
    3. Evaluate tab  → upload annotated TSV, run, view F1 + error rows
    4. Prompts tab   → edit SS / INT prompts inline, save, re-run
    5. Single Turn   → quickly test one turn before a full batch run
    6. History tab   → track F1 across prompt iterations
    7. Export tab    → download final refined_prompts.py for SageMaker
"""

import os
import re
import sys
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omaha_sagemaker as om

st.set_page_config(
    page_title="Omaha Mapper",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialisation ──────────────────────────────────────────────

def _init():
    defaults = {
        "api_key":        os.environ.get("OPENAI_API_KEY", ""),
        "ss_prompt":      om.SS_PROMPT_TEMPLATE,
        "int_prompt":     om.INTERVENTION_PROMPT_TEMPLATE,
        "metrics":        None,
        "df_out":         None,
        "omaha_path":     None,
        "history":        [],   # list of dicts: {run, model, top_k, verify, ss_f1, int_f1}
        "run_count":      0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    api_input = st.text_input(
        "OpenAI API Key",
        value=st.session_state.api_key,
        type="password",
        help="Stored in this browser session only. Never sent to any server except OpenAI.",
    )
    if api_input != st.session_state.api_key:
        st.session_state.api_key = api_input
    if st.session_state.api_key:
        os.environ["OPENAI_API_KEY"] = st.session_state.api_key
        st.success("API key ready ✓")
    else:
        st.warning("Enter your OpenAI API key to run evaluation.")

    st.divider()

    model_name = st.selectbox("Model", list(om.LLM_CONFIGS.keys()),
                               help="gpt-4o gives highest accuracy; gpt-4o-mini is faster/cheaper.")
    top_k      = st.slider("Retrieval top-K", min_value=5, max_value=30, value=15,
                            help="Number of Omaha options retrieved per turn.")
    use_verify = st.checkbox(
        "Verification agent (2nd LLM pass)",
        value=False,
        help="Runs a second LLM call per prediction to confirm or correct it. ~2× cost.",
    )

    st.divider()

    st.markdown("**Omaha Excel**")
    omaha_file = st.file_uploader(
        "Upload Omaha System Excel (.xlsx)", type=["xlsx"], key="omaha_xl",
        help="Upload once — the embedding index is cached for the session.",
    )
    if omaha_file:
        tmp_path = f"/tmp/omaha_{omaha_file.name}"
        with open(tmp_path, "wb") as f:
            f.write(omaha_file.read())
        if tmp_path != st.session_state.omaha_path:
            st.session_state.omaha_path = tmp_path
            _build_index.clear()   # invalidate cached index when new file loaded
        st.success(f"Loaded: {omaha_file.name}")


# ── Cached embedding index ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Building embedding index (first run only)…")
def _build_index(omaha_path: str):
    from sentence_transformers import SentenceTransformer
    sheets   = pd.read_excel(omaha_path, sheet_name=None, engine="openpyxl")
    names    = list(sheets.keys())
    ss_df    = sheets[names[0]].dropna(how="all").reset_index(drop=True)
    int_df   = sheets[names[1]].dropna(how="all").reset_index(drop=True)
    ss_df.columns  = [c.strip() for c in ss_df.columns]
    int_df.columns = [c.strip() for c in int_df.columns]
    embed    = SentenceTransformer(om.EMBEDDING_MODEL, device="cpu")
    ss_docs  = om.build_ss_documents(ss_df)
    int_docs = om.build_intervention_documents(int_df)
    ss_emb   = om.build_index(ss_docs,  embed)
    int_emb  = om.build_index(int_docs, embed)
    return ss_docs, ss_emb, int_docs, int_emb, embed


def _resources():
    if not st.session_state.omaha_path:
        st.error("Upload the Omaha Excel in the sidebar first.")
        return None
    return _build_index(st.session_state.omaha_path)


def _apply_prompts():
    """Push session-state prompts into the om module before any inference."""
    om.SS_PROMPT_TEMPLATE           = st.session_state.ss_prompt
    om.INTERVENTION_PROMPT_TEMPLATE = st.session_state.int_prompt


# ── Helper: build error-analysis DataFrame ─────────────────────────────────────

def _error_table(df_out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df_out.iterrows():
        human_ss  = [str(row.get(f"OS_SS_{j}", "")).strip() for j in range(1, 4)
                     if str(row.get(f"OS_SS_{j}", "")).strip() not in ("", "nan")]
        pred_ss   = [str(row.get(f"Pred_SS_{j}", "")).strip() for j in range(1, 4)
                     if str(row.get(f"Pred_SS_{j}", "")).strip() not in ("", "nan")]
        human_int = [str(row.get(f"OS_I_{j}", "")).strip() for j in range(1, 4)
                     if str(row.get(f"OS_I_{j}", "")).strip() not in ("", "nan")]
        pred_int  = [str(row.get(f"Pred_I_{j}", "")).strip() for j in range(1, 4)
                     if str(row.get(f"Pred_I_{j}", "")).strip() not in ("", "nan")]

        ss_tp, ss_fp, ss_fn  = om.row_level_match(pred_ss,  human_ss)
        it_tp, it_fp, it_fn  = om.row_level_match(pred_int, human_int)

        if ss_fp or ss_fn or it_fp or it_fn:
            rows.append({
                "Turn (first 130 chars)": str(row.get("Conversation", ""))[:130],
                "SS expected":   " | ".join(human_ss)  or "—",
                "SS predicted":  " | ".join(pred_ss)   or "—",
                "SS error":      ("FN " if ss_fn else "") + ("FP" if ss_fp else ""),
                "INT expected":  " | ".join(human_int) or "—",
                "INT predicted": " | ".join(pred_int)  or "—",
                "INT error":     ("FN " if it_fn else "") + ("FP" if it_fp else ""),
            })
    return pd.DataFrame(rows)


def _show_metrics(m: dict, label: str = ""):
    ss_p, ss_r, ss_f1 = om.compute_prf(m["ss_tp"],  m["ss_fp"],  m["ss_fn"])
    i_p,  i_r,  i_f1  = om.compute_prf(m["int_tp"], m["int_fp"], m["int_fn"])

    def _delta(f):
        return "✓ ≥ 0.90" if f >= 0.9 else f"gap {f - 0.9:+.3f}"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("SS F1",         f"{ss_f1:.3f}", _delta(ss_f1))
    c2.metric("SS Precision",  f"{ss_p:.3f}")
    c3.metric("SS Recall",     f"{ss_r:.3f}")
    c4.metric("INT F1",        f"{i_f1:.3f}", _delta(i_f1))
    c5.metric("INT Precision", f"{i_p:.3f}")
    c6.metric("INT Recall",    f"{i_r:.3f}")

    if ss_f1 >= 0.9 and i_f1 >= 0.9:
        st.success("🎯 Both SS F1 and INT F1 ≥ 0.90 — target reached!")
    else:
        gaps = []
        if ss_f1 < 0.9: gaps.append(f"SS: {0.9 - ss_f1:.3f}")
        if i_f1  < 0.9: gaps.append(f"INT: {0.9 - i_f1:.3f}")
        st.warning(f"Remaining gap → {', '.join(gaps)}")

    return ss_f1, i_f1


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_eval, tab_prompts, tab_single, tab_history, tab_export = st.tabs([
    "📊 Evaluate", "✏️ Prompts", "🔬 Single Turn", "📈 History", "📦 Export",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Batch Evaluate
# ══════════════════════════════════════════════════════════════════════════════

with tab_eval:
    st.header("Batch Evaluation")
    st.caption(
        "Upload your annotated TSV transcript, run inference, and see "
        "F1 scores + error rows.  Edit prompts in the **Prompts** tab and re-run."
    )

    tsv_file = st.file_uploader(
        "Annotated TSV transcript (with OS_SS_1/2/3 and OS_I_1/2/3 columns)",
        type=["tsv", "txt", "csv"],
        key="tsv_upload",
    )

    col_btn, col_opt = st.columns([1, 4])
    with col_btn:
        run_btn = st.button(
            "▶ Run Evaluation", type="primary",
            disabled=not (st.session_state.api_key
                          and st.session_state.omaha_path
                          and tsv_file is not None),
        )
    with col_opt:
        show_errors = st.checkbox("Show error rows below results", value=True)

    if run_btn:
        res = _resources()
        if res:
            ss_docs, ss_emb, int_docs, int_emb, embed = res
            _apply_prompts()

            df = pd.read_csv(tsv_file, sep="\t", dtype=str).fillna("")
            df.columns = [c.strip() for c in df.columns]

            orig_k = om.TOP_K_RETRIEVAL
            om.TOP_K_RETRIEVAL = top_k

            with st.spinner(f"Running {len(df)} rows with {model_name}…"):
                df_out = om.process_sheet(
                    df, ss_docs, ss_emb, int_docs, int_emb, embed, model_name
                )

            if use_verify:
                with st.spinner("Running verification agent (2nd LLM pass)…"):
                    df_out = om.verify_sheet(
                        df_out, ss_docs, ss_emb, int_docs, int_emb,
                        embed, model_name, top_k
                    )

            om.TOP_K_RETRIEVAL = orig_k

            metrics = om.evaluate_sheet(df_out)
            st.session_state.metrics = metrics
            st.session_state.df_out  = df_out

            ss_p, ss_r, ss_f1 = om.compute_prf(metrics["ss_tp"],  metrics["ss_fp"],  metrics["ss_fn"])
            i_p,  i_r,  i_f1  = om.compute_prf(metrics["int_tp"], metrics["int_fp"], metrics["int_fn"])

            st.session_state.run_count += 1
            st.session_state.history.append({
                "Run":     st.session_state.run_count,
                "Model":   model_name,
                "Top-K":   top_k,
                "Verify":  use_verify,
                "SS F1":   round(ss_f1, 4),
                "INT F1":  round(i_f1,  4),
                "SS prompt (first 60)":  st.session_state.ss_prompt.strip()[:60],
                "INT prompt (first 60)": st.session_state.int_prompt.strip()[:60],
            })

    if st.session_state.metrics:
        st.divider()
        ss_f1, i_f1 = _show_metrics(st.session_state.metrics)

        if show_errors and st.session_state.df_out is not None:
            err_df = _error_table(st.session_state.df_out)
            if len(err_df):
                st.subheader(f"Error rows ({len(err_df)})")
                st.caption(
                    "FN = missed label (model said None or wrong).  "
                    "FP = extra label (model added something not in ground truth)."
                )
                st.dataframe(err_df, use_container_width=True, height=380)
                st.download_button(
                    "Download error rows as CSV",
                    data=err_df.to_csv(index=False),
                    file_name="errors.csv",
                    mime="text/csv",
                )
            else:
                st.success("No errors — perfect match on all rows!")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Edit Prompts
# ══════════════════════════════════════════════════════════════════════════════

with tab_prompts:
    st.header("Edit Prompts")
    st.caption(
        "Modify the SS and/or INT prompt below, click **Save**, "
        "then go to **Evaluate** and re-run to see the effect."
    )

    load_c1, load_c2, _ = st.columns([1, 1, 3])
    with load_c1:
        if st.button("Load refined_prompts.py"):
            try:
                from refined_prompts import MY_SS_PROMPT, MY_INT_PROMPT
                st.session_state.ss_prompt  = MY_SS_PROMPT
                st.session_state.int_prompt = MY_INT_PROMPT
                st.rerun()
            except ImportError:
                st.error("refined_prompts.py not found in the same directory.")
    with load_c2:
        if st.button("Reset → baseline"):
            st.session_state.ss_prompt  = om.SS_PROMPT_TEMPLATE
            st.session_state.int_prompt = om.INTERVENTION_PROMPT_TEMPLATE
            st.rerun()

    col_ss, col_int = st.columns(2)

    with col_ss:
        st.subheader("Signs / Symptoms Prompt")
        new_ss = st.text_area(
            "ss_prompt_area",
            value=st.session_state.ss_prompt,
            height=650,
            label_visibility="collapsed",
        )
        if st.button("💾 Save SS Prompt"):
            st.session_state.ss_prompt = new_ss
            st.success("SS prompt saved. Re-run Evaluate to apply.")

    with col_int:
        st.subheader("Intervention Prompt")
        new_int = st.text_area(
            "int_prompt_area",
            value=st.session_state.int_prompt,
            height=650,
            label_visibility="collapsed",
        )
        if st.button("💾 Save INT Prompt"):
            st.session_state.int_prompt = new_int
            st.success("INT prompt saved. Re-run Evaluate to apply.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Single Turn Test
# ══════════════════════════════════════════════════════════════════════════════

with tab_single:
    st.header("Test a Single Turn")
    st.caption(
        "Quickly classify one turn with your current prompts — no full batch needed. "
        "Great for testing a prompt fix before running the whole transcript."
    )

    turn_input = st.text_area(
        "Conversation turn",
        height=90,
        placeholder="Type or paste one conversation turn here…",
    )
    ctx_input = st.text_area(
        "Context / surrounding turns (optional)",
        height=70,
        placeholder="Paste a few surrounding turns to improve intervention classification…",
    )

    classify_btn = st.button(
        "Classify",
        disabled=not (st.session_state.api_key and bool(turn_input.strip())),
    )

    if classify_btn:
        res = _resources()
        if res:
            ss_docs, ss_emb, int_docs, int_emb, embed = res
            _apply_prompts()

            orig_k = om.TOP_K_RETRIEVAL
            om.TOP_K_RETRIEVAL = top_k

            with st.spinner("Classifying…"):
                # SS
                ss_retrieved = om.retrieve(turn_input, ss_docs, ss_emb, embed)
                ss_prompt    = om.build_ss_prompt(turn_input, ctx_input, ss_retrieved)
                if om.LLM_CONFIGS[model_name]["provider"] != "local":
                    ss_prompt = re.sub(r"<\|[^|>]+\|>", "", ss_prompt).strip()
                ss_raw   = om.call_llm(ss_prompt, model_name)
                ss_preds = om.parse_ss_output(ss_raw)

                # INT
                int_retrieved = om.retrieve(turn_input, int_docs, int_emb, embed)
                int_prompt    = om.build_intervention_prompt(turn_input, ctx_input, int_retrieved)
                int_raw   = om.call_llm(int_prompt, model_name)
                int_preds = om.parse_intervention_output(int_raw)

                # Optional verification
                if use_verify and ss_preds:
                    ss_preds = om._verify_ss(
                        turn_input, ss_preds, ss_docs, ss_emb, embed, model_name
                    )
                if use_verify and int_preds:
                    int_preds = om._verify_int(
                        turn_input, ctx_input, int_preds, int_docs, int_emb, embed, model_name
                    )

            om.TOP_K_RETRIEVAL = orig_k

            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("Signs / Symptoms")
                if ss_preds:
                    for p in ss_preds:
                        st.info(
                            f"**{p.get('problem', '?')}** — {p.get('ss', '?')}\n\n"
                            f"_{p.get('domain', '')}_"
                        )
                else:
                    st.info("None")
                with st.expander("Raw LLM response (SS)"):
                    st.text(ss_raw)
                with st.expander("Retrieved options (SS)"):
                    for i, d in enumerate(ss_retrieved, 1):
                        st.text(f"{i}. {d['domain']} | {d['problem']} | {d['ss']}")

            with col_b:
                st.subheader("Interventions")
                if int_preds:
                    for p in int_preds:
                        st.info(f"**{p.get('category', '?')}** | {p.get('target', '?')}")
                else:
                    st.info("None")
                with st.expander("Raw LLM response (INT)"):
                    st.text(int_raw)
                with st.expander("Retrieved options (INT)"):
                    for i, d in enumerate(int_retrieved, 1):
                        st.text(f"{i}. {d['category']} | {d['target']}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — History
# ══════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.header("Evaluation History")
    st.caption("Track F1 across prompt iterations to see what is improving.")

    hist = st.session_state.history
    if hist:
        hist_df = pd.DataFrame(hist).set_index("Run")
        st.dataframe(
            hist_df[["Model", "Top-K", "Verify", "SS F1", "INT F1"]],
            use_container_width=True,
        )

        chart_df = hist_df[["SS F1", "INT F1"]].copy()
        st.line_chart(chart_df, use_container_width=True)

        st.caption("Dashed line at 0.90 (target)")

        col_dl, col_clr = st.columns([1, 1])
        with col_dl:
            st.download_button(
                "Download history as CSV",
                data=hist_df.reset_index().to_csv(index=False),
                file_name="eval_history.csv",
                mime="text/csv",
            )
        with col_clr:
            if st.button("Clear history"):
                st.session_state.history   = []
                st.session_state.run_count = 0
                st.rerun()
    else:
        st.info("No evaluation runs yet. Use the **Evaluate** tab first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Export
# ══════════════════════════════════════════════════════════════════════════════

with tab_export:
    st.header("Export for SageMaker")
    st.caption(
        "Download your current prompts as **refined_prompts.py**.  "
        "Drop that file into your SageMaker repo and run:\n\n"
        "```\npython eval_local.py --prompts refined\n```"
    )

    # Escape triple-quotes inside the prompts to avoid breaking the f-string
    ss_safe  = st.session_state.ss_prompt.replace('"""', "'''")
    int_safe = st.session_state.int_prompt.replace('"""', "'''")

    py_content = f'''\"""
Refined Omaha System prompt templates.
Generated by the Omaha Mapper interactive app.

Usage:
    python eval_local.py --prompts refined [--verbose]
\"""

MY_SS_PROMPT = """{ss_safe}"""

MY_INT_PROMPT = """{int_safe}"""
'''

    st.download_button(
        "⬇️ Download refined_prompts.py",
        data=py_content,
        file_name="refined_prompts.py",
        mime="text/x-python",
    )

    with st.expander("Preview (first 3 000 chars)"):
        st.code(py_content[:3000] + ("…" if len(py_content) > 3000 else ""),
                language="python")

    st.divider()
    st.markdown("**How to use in SageMaker after downloading:**")
    st.code(
        "# 1. Copy refined_prompts.py to your omaha_mapping/ folder\n"
        "# 2. Run evaluation with refined prompts\n"
        "python eval_local.py --prompts refined --verbose\n\n"
        "# 3. Or import directly in your pipeline script\n"
        "from refined_prompts import MY_SS_PROMPT, MY_INT_PROMPT\n"
        "import omaha_sagemaker as om\n"
        "om.SS_PROMPT_TEMPLATE          = MY_SS_PROMPT\n"
        "om.INTERVENTION_PROMPT_TEMPLATE = MY_INT_PROMPT",
        language="bash",
    )
