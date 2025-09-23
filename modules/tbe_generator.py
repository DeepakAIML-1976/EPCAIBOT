# modules/tbe_generator.py
import re
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from modules import embedding_handler as eh


# =========================
# Helpers: text collection
# =========================
def _collect_text_for_source(
    source_name: str, namespace: str, model_choice: str = "openai", top_k: int = 300
) -> str:
    """
    Collect and concatenate chunks indexed under a given source filename.
    Filters strictly by filename/source to avoid noise.
    """
    hits = eh.search(source_name, top_k=top_k, model_choice=model_choice, namespace=namespace)
    filtered = [h for h in hits if h.get("source") == source_name or h.get("filename") == source_name]
    try:
        filtered.sort(key=lambda x: int(x.get("chunk_id", 0)))
    except Exception:
        pass
    return "\n".join(h.get("text", "") for h in filtered).strip()


# ==========================================
# Normalize and fuzzy key matching utilities
# ==========================================
def _normalize_key(k: str) -> str:
    if not k:
        return ""
    k = k.lower().strip()
    k = re.sub(r"[^a-z0-9\s/().:%+-]", " ", k)
    k = re.sub(r"\s+", " ", k)
    return k


def _best_match_key(ds_key: str, vendor_keys: List[str]) -> Tuple[str, float]:
    """
    Return best vendor key and score using token overlap + simple LCS.
    """
    nk = _normalize_key(ds_key)
    if not nk:
        return "", 0.0

    ds_tokens = set(nk.split())
    best_key, best_score = "", 0.0

    def lcs_len(a: str, b: str) -> int:
        la, lb = len(a), len(b)
        dp = [[0] * (lb + 1) for _ in range(la + 1)]
        best = 0
        for i in range(la - 1, -1, -1):
            for j in range(lb - 1, -1, -1):
                if a[i] == b[j]:
                    dp[i][j] = 1 + dp[i + 1][j + 1]
                    best = max(best, dp[i][j])
        return best

    for vk in vendor_keys:
        nv = _normalize_key(vk)
        if not nv:
            continue
        if nv == nk:
            return vk, 1.0
        vtoks = set(nv.split())
        overlap = len(ds_tokens & vtoks) / max(1, len(ds_tokens | vtoks))
        lcs_ratio = lcs_len(nk, nv) / max(1, min(len(nk), len(nv)))
        score = max(overlap, 0.8 * lcs_ratio)
        if score > best_score:
            best_key, best_score = vk, score
    return best_key, best_score


# ==========================================
# Parsers (KV, pipe tables, whitespace tables)
# ==========================================
KV_SEPARATORS = (":", " — ", " - ")


def _parse_kv_pairs(text: str) -> List[Tuple[str, str]]:
    """
    Heuristic KV extractor: 'Key: Value', 'Key - Value', 'Key | Value', headings + short next line.
    """
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    kvs: List[Tuple[str, str]] = []

    # 1) explicit separators
    for ln in lines:
        # pipes
        if "|" in ln and not ln.startswith("---"):
            parts = [p.strip() for p in ln.split("|") if p.strip()]
            if len(parts) == 2:
                kvs.append((parts[0], parts[1]))
                continue
            if len(parts) > 2 and len(parts[0].split()) <= 6:
                kvs.append((parts[0], " | ".join(parts[1:])))
                continue

        # colon / dash
        for sep in KV_SEPARATORS:
            if sep in ln:
                left, right = ln.split(sep, 1)
                left, right = left.strip(), right.strip()
                if left and len(left.split()) <= 12:
                    kvs.append((left, right if right else "Not specified"))
                    break

    if kvs:
        # Deduplicate by normalized key (keep first)
        seen = set()
        out = []
        for k, v in kvs:
            nk = _normalize_key(k)
            if nk and nk not in seen:
                seen.add(nk)
                out.append((k, v))
        return out

    # 2) headings + next line
    out = []
    i = 0
    while i < len(lines) - 1:
        head, val = lines[i], lines[i + 1]
        if len(head.split()) <= 6 and len(val.split()) <= 30:
            out.append((head, val))
            i += 2
        else:
            i += 1
    return out


def _parse_pipe_table(text: str) -> List[List[str]]:
    """
    Parse Markdown-like tables composed of pipe-delimited rows.
    Returns list of rows (each a list of cells), already stripped.
    """
    rows = []
    for ln in text.splitlines():
        if ln.strip().startswith("|") and "|" in ln.strip()[1:]:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            # ignore alignment/separator rows like |---|---|
            if all(set(c) <= {"-", ":"} for c in cells):
                continue
            rows.append(cells)
    return rows


def _parse_whitespace_columns(text: str, min_cols: int = 2) -> List[List[str]]:
    """
    Parse rows where columns are separated by >=2 spaces or tabs.
    Returns list of rows.
    """
    rows = []
    for ln in text.splitlines():
        if not ln.strip():
            continue
        if re.search(r"\s{2,}|\t", ln):
            parts = re.split(r"\s{2,}|\t", ln.strip())
            parts = [p.strip() for p in parts if p is not None]
            if len(parts) >= min_cols:
                rows.append(parts)
    return rows


# ==========================================
# Vendor mapping & DataFrame construction
# ==========================================
def _extract_vendor_maps(vendor_docs: List[str], model_choice: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """
    For each vendor doc, build a KV map using KV parser OR convert first two columns of any table
    into KV when that's clearly 'Requirement | Value'.
    """
    vendor_maps: Dict[str, Dict[str, str]] = {}
    vendor_names: Dict[str, str] = {}

    for vd in vendor_docs:
        vtext = _collect_text_for_source(vd, "vendor_docs", model_choice=model_choice)

        # 1) Try KV first
        pairs = _parse_kv_pairs(vtext)

        # 2) If no KV, try pipe-table then whitespace-table, assume first col is key, second is value
        if not pairs:
            pipe_rows = _parse_pipe_table(vtext)
            if pipe_rows:
                # skip header if looks like header
                start = 1 if pipe_rows and any(h.lower() in ("requirement", "parameter") for h in pipe_rows[0]) else 0
                for r in pipe_rows[start:]:
                    if len(r) >= 2:
                        pairs.append((r[0], r[1]))

        if not pairs:
            ws_rows = _parse_whitespace_columns(vtext, min_cols=2)
            if ws_rows:
                for r in ws_rows:
                    if len(r) >= 2 and len(r[0].split()) <= 12:
                        pairs.append((r[0], r[1]))

        vendor_maps[vd] = {k: v for k, v in pairs}
        vendor_names[vd] = eh.get_vendor_display_name(vd)

    return vendor_maps, vendor_names


def _build_from_requirements(
    ds_kv: List[Tuple[str, str]], vendor_maps: Dict[str, Dict[str, str]], vendor_names: Dict[str, str], vendor_docs: List[str]
) -> pd.DataFrame:
    """Use datasheet requirements list; fill vendor columns by best-match against vendor maps."""
    rows = []
    for ds_k, ds_v in ds_kv:
        row = {"Requirement": ds_k, "Datasheet": ds_v if ds_v else "Not specified"}
        for vd in vendor_docs:
            vmap = vendor_maps.get(vd, {})
            # try exact normalized
            matched_val = None
            for vk in vmap.keys():
                if _normalize_key(vk) == _normalize_key(ds_k):
                    matched_val = vmap[vk]
                    break
            # fuzzy if needed
            if matched_val is None and vmap:
                best_vk, score = _best_match_key(ds_k, list(vmap.keys()))
                if score >= 0.35:
                    matched_val = vmap.get(best_vk)
            if not matched_val:
                matched_val = "Not specified"
            row[vendor_names.get(vd, vd)] = matched_val
        rows.append(row)

    df = pd.DataFrame(rows)
    # stabilize columns
    vendor_cols = [vendor_names[vd] for vd in vendor_docs]
    for c in ["Requirement", "Datasheet", *vendor_cols]:
        if c not in df.columns:
            df[c] = "Not specified"
    return df[["Requirement", "Datasheet", *vendor_cols]]


def _build_from_tabular_lines(
    ds_text: str, vendor_docs: List[str], model_choice: str, vendor_names: Dict[str, str]
) -> pd.DataFrame:
    """
    If the underlying datasheet text itself is a table, build rows directly from its table rows:
    - Prefer pipe table, then whitespace table.
    """
    rows = _parse_pipe_table(ds_text)
    if not rows:
        rows = _parse_whitespace_columns(ds_text, min_cols=2)

    if rows and any(len(r) >= 2 for r in rows):
        header = rows[0]
        body = rows[1:] if any(h.lower() in ("requirement", "datasheet") for h in header) else rows

        vendor_cols = [vendor_names.get(vd, vd) for vd in vendor_docs]
        normalized = []
        for r in body:
            if len(r) < 2:
                continue
            req = r[0]
            ds_val = r[1]
            row = {"Requirement": req, "Datasheet": ds_val}
            for idx, vcol in enumerate(vendor_cols):
                row[vcol] = r[2 + idx] if len(r) > 2 + idx else "Not specified"
            normalized.append(row)
        if normalized:
            df = pd.DataFrame(normalized)
            for vcol in vendor_cols:
                if vcol not in df.columns:
                    df[vcol] = "Not specified"
            return df[["Requirement", "Datasheet", *vendor_cols]]

    return pd.DataFrame(columns=["Requirement", "Datasheet", *[vendor_names.get(vd, vd) for vd in vendor_docs]])


def _build_comparison_df(datasheet_name: str, vendor_docs: List[str], model_choice: str = "openai") -> pd.DataFrame:
    """
    Build the TBE table with multiple robust fallbacks.
    """
    ds_text = _collect_text_for_source(datasheet_name, "datasheets", model_choice=model_choice)

    # 1) Datasheet as KV list of requirements
    ds_kv = _parse_kv_pairs(ds_text)

    vendor_maps, vendor_names = _extract_vendor_maps(vendor_docs, model_choice=model_choice)

    if ds_kv:
        df = _build_from_requirements(ds_kv, vendor_maps, vendor_names, vendor_docs)
        if not df.empty:
            return df

    # 2) If no KV on datasheet, try to build from tabular lines in the datasheet text itself
    df_tab = _build_from_tabular_lines(ds_text, vendor_docs, model_choice, vendor_names)
    if not df_tab.empty:
        return df_tab

    # 3) Last fallback: synthesize requirements from vendor keys union
    all_keys = set()
    for vmap in vendor_maps.values():
        all_keys.update(vmap.keys())
    ds_kv_fallback = [(k, "Not specified") for k in list(all_keys)[:50]]
    if ds_kv_fallback:
        return _build_from_requirements(ds_kv_fallback, vendor_maps, vendor_names, vendor_docs)

    return pd.DataFrame(columns=["Requirement", "Datasheet", *[vendor_names.get(vd, vd) for vd in vendor_docs]])


# =========================
# HTML Grid renderer for Streamlit
# =========================
def render_tbe_html_grid(df: pd.DataFrame):
    """
    Render the TBE DataFrame as an HTML grid in Streamlit with inline CSS styling.
    First column: Requirement
    Second column: Datasheet value
    Following columns: vendor values (with color-coded cells)
    """
    if df.empty:
        st.warning("No data to render in TBE.")
        return

    req_col = "Requirement"
    ds_col = "Datasheet"
    vendor_cols = [c for c in df.columns if c not in (req_col, ds_col)]

    # Build HTML
    html_parts = []
    html_parts.append(
        """
        <style>
        table.tbe-grid {
            border-collapse: collapse;
            width: 100%;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
            font-size: 14px;
        }
        table.tbe-grid th, table.tbe-grid td {
            border: 1px solid #e0e0e0;
            padding: 8px 10px;
            vertical-align: top;
            text-align: left;
        }
        table.tbe-grid th {
            background-color: #f7f7f7;
            font-weight: 600;
        }
        .cell-red { background-color: #FFC7CE; }   /* missing */
        .cell-green { background-color: #C6EFCE; } /* match */
        .cell-yellow { background-color: #FFEB9C; } /* different */
        .req-col { width: 28%; font-weight: 600; }
        .ds-col { width: 28%; }
        .vendor-col { width: auto; }
        </style>
        """
    )

    # Header
    html_parts.append("<table class='tbe-grid'><thead><tr>")
    html_parts.append(f"<th class='req-col'>{req_col}</th>")
    html_parts.append(f"<th class='ds-col'>{ds_col}</th>")
    for v in vendor_cols:
        html_parts.append(f"<th class='vendor-col'>{v}</th>")
    html_parts.append("</tr></thead><tbody>")

    # Rows
    for _, r in df.iterrows():
        html_parts.append("<tr>")
        # Requirement cell
        req_html = str(r.get(req_col, ""))
        html_parts.append(f"<td class='req-col'>{req_html}</td>")
        # Datasheet cell
        ds_val = str(r.get(ds_col, "")).strip()
        ds_norm = ds_val.lower()
        html_parts.append(f"<td class='ds-col'>{ds_val}</td>")
        # Vendor cells
        for v in vendor_cols:
            vval = str(r.get(v, "")).strip()
            vnorm = vval.lower()
            if not vnorm or vnorm == "not specified":
                cls = "cell-red"
            elif ds_norm and vnorm == ds_norm:
                cls = "cell-green"
            else:
                cls = "cell-yellow"
            # Escape HTML characters minimally
            v_html = (
                vval.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br/>")
            )
            html_parts.append(f"<td class='{cls} vendor-col'>{v_html}</td>")
        html_parts.append("</tr>")

    html_parts.append("</tbody></table>")

    html = "\n".join(html_parts)
    st.markdown(html, unsafe_allow_html=True)


# =========================
# Streamlit UI
# =========================
def tbe_ui():
    st.header("📊 Technical Bid Evaluation (TBE) – HTML Grid Matrix")
    model_choice = st.selectbox("Select embedding model", ["openai", "scibert", "matscibert"])
    datasheets = eh.get_datasheets()
    if not datasheets:
        st.info("Please upload datasheets and vendor offers first.")
        return

    ds = st.selectbox("Choose datasheet", datasheets, key=f"tbe_ds_select_{hash(tuple(datasheets))}")
    vendor_docs = eh.get_vendors_for_datasheet(ds)
    if not vendor_docs:
        st.info("No vendor offers linked to this datasheet yet.")
        return

    st.markdown("**Vendors linked:** " + ", ".join([eh.get_vendor_display_name(vd) for vd in vendor_docs]))

    show_debug = st.checkbox("Show debug previews", value=False, key=f"tbe_debug_{ds}")
    if show_debug:
        ds_text_preview = _collect_text_for_source(ds, "datasheets", model_choice=model_choice)[:1200]
        st.code(f"[Datasheet first 1200 chars]\n{ds_text_preview}")
        for vd in vendor_docs:
            vtext_preview = _collect_text_for_source(vd, "vendor_docs", model_choice=model_choice)[:1200]
            st.code(f"[Vendor {eh.get_vendor_display_name(vd)} first 1200 chars]\n{vtext_preview}")

    if st.button("Generate HTML Grid TBE", key=f"btn_gen_tbe_html_{ds}"):
        with st.spinner("Building comparison matrix..."):
            df = _build_comparison_df(ds, vendor_docs, model_choice=model_choice)
            st.subheader(f"TBE – {ds}")
            if df.empty:
                st.warning("No rows could be parsed. Try enabling 'Show debug previews' and verify text extraction.")
                return
            render_tbe_html_grid(df)
