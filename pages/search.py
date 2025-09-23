import streamlit as st
from modules.embedding_store import search_embedding
from modules.tq_generator import generate_tq

def app():
    st.header("Live Embedding Search & TQ Drafting")
    
    query = st.text_area("Enter requirement, partial TQ, or vendor detail")
    top_k = st.slider("Number of similar results", 1, 10, 5)

    if st.button("Search"):
        results = search_embedding(query, top_k=top_k)
        if results:
            st.success(f"Found {len(results)} similar entries")
            for i, r in enumerate(results):
                st.markdown(f"**Result {i+1}**")
                st.write(f"Text: {r['text']}")
                st.write(f"Source: {r['metadata']}")
                st.write(f"Similarity Score: {r['score']:.4f}")

                if st.button(f"Generate GPT-4 TQ from Result {i+1}", key=i):
                    tq = generate_tq(r['text'])
                    st.markdown("**Generated Technical Query (TQ):**")
                    st.write(tq)
                st.markdown("---")
        else:
            st.warning("No similar results found.")
