from typing import List, Sequence, Tuple, Optional, Dict, Union, Callable
from streamlit import sidebar, header as stHeader, columns as stColumns, info as stInfo, write as stWrite, text_input as stTextarea, columns as stColumns, expander as stExpander, dataframe as stDataframe #as st 
from spacy.tokens import Doc as spcyDoc
from spacy.language import Language
from spacy import displacy
from pandas import DataFrame

from utils import loadMdl, prcssTxt, get_svg, get_html, get_color_styles, LOGO


# fmt: off
NER_ATTRS = ["text", "label_", "start", "end", "start_char", "end_char"]
TOKEN_ATTRS = ["idx", "text", "lemma_", "pos_", "tag_", "dep_", "head", "morph", "ent_type_", "ent_iob_", "shape_", "is_alpha", "is_ascii", "is_digit", "is_punct", "like_num", "is_sent_start"]
# fmt: on
FOOTER = """<span style="font-size: 0.75em">&hearts; Built with [`spacy-streamlit`](https://github.com/explosion/spacy-streamlit)</span>"""


def visualize(
    models: Union[List[str], Dict[str, str]],
    default_text: str = "",
    default_model: Optional[str] = None,
    visualizers: List[str] = ["parser", "ner", "textcat", "similarity", "tokens"],
    ner_labels: Optional[List[str]] = None,
    ner_attrs: List[str] = NER_ATTRS,
    similarity_texts: Tuple[str, str] = ("apple", "orange"),
    token_attrs: List[str] = TOKEN_ATTRS,
    show_json_doc: bool = True,
    show_meta: bool = True,
    show_config: bool = True,
    show_visualizer_select: bool = False,
    show_pipeline_info: bool = True,
    sidebar_title: Optional[str] = None,
    sidebar_description: Optional[str] = None,
    show_logo: bool = True,
    color: Optional[str] = "#09A3D5",
    key: Optional[str] = None,
    get_default_text: Callable[[Language], str] = None,
) -> None:
    """Embed the full visualizer with selected components."""
    if color:
        stWrite(get_color_styles(color), unsafe_allow_html=True)
    if show_logo:
        sidebar.markdown(LOGO, unsafe_allow_html=True)
    if sidebar_title:
        sidebar.title(sidebar_title)
    if sidebar_description:
        sidebar.markdown(sidebar_description)

    # Allow both dict of model name / description as well as list of names
    model_names = models
    format_func = str
    if isinstance(models, dict):
        format_func = lambda name: models.get(name, name)
        model_names = list(models.keys())

    default_model_index = (
        model_names.index(default_model)
        if default_model is not None and default_model in model_names
        else 0
    )
    #spacy_model = sidebar.selectbox( "Model", model_names, index=default_model_index, key=f"{key}_visualize_models", format_func=format_func)
    #model_load_state = stInfo(f"Loading model '{spacy_model}'...")
    from utils import loadMdl
    from streamlit import session_state
    spacy_model=session_state['medModel']
    nlp = loadMdl(spacy_model)
    model_load_state.empty()

    if show_pipeline_info:
      sidebar.subheader("Pipeline info")
      desc = f"""<p style="font-size: 0.85em; line-height: 1.5"><strong>{spacy_model}:</strong> <code>v{nlp.meta['version']}</code>. {nlp.meta.get("description", "")}</p>"""
      sidebar.markdown(desc, unsafe_allow_html=True)

    if show_visualizer_select:
      'active_visualizers'
      #active_visualizers = sidebar.multiselect("Visualizers", options=visualizers, default=list(visualizers), key=f"{key}_viz_select")
    else: active_visualizers = visualizers

    default_text = (get_default_text(nlp) if get_default_text is not None else default_text)
    text = stTextarea("Text to analyze", default_text, key=f"{key}_visualize_text")
    doc = prcssTxt(spacy_model, text)

    if "parser" in visualizers and "parser" in active_visualizers:
        visualize_parser(doc, key=key)
    if "ner" in visualizers and "ner" in active_visualizers:
        ner_labels = ner_labels or nlp.get_pipe("ner").labels
        visualize_ner(doc, labels=ner_labels, attrs=ner_attrs, key=key)
    if "textcat" in visualizers and "textcat" in active_visualizers:
        visualize_textcat(doc)
    if "similarity" in visualizers and "similarity" in active_visualizers:
        visualize_similarity(nlp, key=key)
    if "tokens" in visualizers and "tokens" in active_visualizers:
        visualize_tokens(doc, attrs=token_attrs, key=key)

    if show_json_doc or show_meta or show_config:
        st.header("Pipeline information")
        if show_json_doc:
            json_doc_exp = st.beta_expander("JSON Doc")
            json_doc_exp.json(doc.to_json())

        if show_meta:
            meta_exp = st.beta_expander("Pipeline meta.json")
            meta_exp.json(nlp.meta)

        if show_config:
            config_exp = st.beta_expander("Pipeline config.cfg")
            config_exp.code(nlp.config.to_str())

    sidebar.markdown( FOOTER, unsafe_allow_html=True,)


def visualize_parser(doc: spcyDoc, *, title: Optional[str] = "Dependency Parse & Part-of-speech tags",
    key: Optional[str] = None,
) -> None:
    """Visualizer for dependency parses."""
    if title:
        stHeader(title)
    cols = stColumns(4)
    split_sents = cols[0].checkbox(
        "Split sentences", value=True, key=f"{key}_parser_split_sents"
    )
    options = {
        "collapse_punct": cols[1].checkbox(
            "Collapse punct", value=True, key=f"{key}_parser_collapse_punct"
        ),
        "collapse_phrases": cols[2].checkbox(
            "Collapse phrases", key=f"{key}_parser_collapse_phrases"
        ),
        "compact": cols[3].checkbox("Compact mode", key=f"{key}_parser_compact"),
    }
    docs = [span.as_doc() for span in doc.sents] if split_sents else [doc]
    for sent in docs:
        html = displacy.render(sent, options=options, style="dep")
        # Double newlines seem to mess with the rendering
        html = html.replace("\n\n", "\n")
        if split_sents and len(docs) > 1:
            st.markdown(f"> {sent.text}")
        st.write(get_svg(html), unsafe_allow_html=True)


def vizNER(doc: spcyDoc, *, labels: Sequence[str] = tuple(), attrs: List[str] = NER_ATTRS, show_table: bool = True, title: Optional[str] = "Named Entities", colors: Dict[str, str] = {}, key: Optional[str] = None) -> None:
    """Visualizer for named entities."""
    #print(doc)
    #print(doc.get_docs())
    for e in doc:print(e.get_docs())
    if title:
        stHeader(title)
    exp = stExpander("Select entity labels")
    label_select = exp.multiselect("Entity labels", options=labels, default=list(labels), key=f"{key}_ner_label_select")
    html = displacy.render( doc, style="ent", options={"ents": label_select, "colors": colors})
    style = "<style>mark.entity { display: inline-block }</style>"
    stWrite(f"{style}{get_html(html)}", unsafe_allow_html=True)
    if show_table:
        data = [ [str(getattr(ent, attr)) for attr in attrs]
            for ent in doc.ents if ent.label_ in labels ]
        df = DataFrame(data, columns=attrs)
        stDataframe(df)

def visualize_textcat( doc: spcyDoc, *, title: Optional[str] = "Text Classification") -> None:
    """Visualizer for text categories."""
    if title:
        st.header(title)
    stMarkdown(f"> {doc.text}")
    df = DataFrame(doc.cats.items(), columns=("Label", "Score"))
    stDataframe(df)

def visualize_similarity(nlp: Language, default_texts: Tuple[str, str] = ("apple", "orange"), *, threshold: float = 0.5, title: Optional[str] = "Vectors & Similarity", key: Optional[str] = None) -> None:
    """Visualizer for semantic similarity using word vectors."""
    meta = nlp.meta.get("vectors", {})
    if title:
        st.header(title)
    if not meta.get("width", 0):
        st.warning("No vectors available in the model.")
    else:
        cols = stColumns(2)
        text1 = cols[0].text_input("Text or word 1", default_texts[0], key=f"{key}_similarity_text1")
        text2 = cols[1].text_input("Text or word 2", default_texts[1], key=f"{key}_similarity_text2")
        doc1 = nlp.make_doc(text1)
        doc2 = nlp.make_doc(text2)
        similarity = doc1.similarity(doc2)
        similarity_text = f"**Score:** `{similarity}`"
        if similarity > threshold: st.success(similarity_text)
        else: st.error(similarity_text)

        exp = stExpander("Vector information")
        exp.code(meta)

def visualize_tokens(doc: spcyDoc, *, attrs: List[str] = TOKEN_ATTRS, title: Optional[str] = "Token attributes", key: Optional[str] = None) -> None:
    """Visualizer for token attributes."""
    if title:
        stHeader(title)
    exp = stExpander("Select token attributes")
    selected = exp.multiselect("Token attributes", options=attrs, default=list(attrs), key=f"{key}_tokens_attr_select")
    data = [[str(getattr(token, attr)) for attr in selected] for token in doc]
    df = DataFrame(data, columns=selected)
    stDataframe(df)
