This sounds less like a RAG issue and more like a Streamlit input state/UI issue.

From what you're describing:

when I press Enter, the query gets sent but the text stays in the input box

and

the screen dims and only the arrow is used

that usually means the input field's value is being preserved in st.session_state and not cleared after submission.

Check your chat input code

If you're using something like:

query = st.text_input(
    "Ask a question",
    key="query_input"
)


then after sending the message you need to clear it:

st.session_state["query_input"] = ""


after processing the query.

If you're using st.chat_input()

Example:

if prompt := st.chat_input("Ask something..."):
    response = rag.answer_question(prompt)


st.chat_input() normally clears itself automatically.

If it doesn't, then you might be:

storing the prompt back into session state
calling st.rerun() at an odd time
rendering a custom text input instead of st.chat_input
Looking at your CSS

I noticed a lot of aggressive overrides:

[data-testid="stTextInput"] input {
    ...
}

[data-testid="stHorizontalBlock"] {
    align-items: center;
}


and

[data-testid="stTextInput"] > div {
    margin-bottom: 0 !important;
}


You're styling text_input, not specifically chat_input.

If your composer is custom-built with:

col1, col2, col3 = st.columns(...)


plus:

st.text_input(...)
st.button(...)


then Enter may submit but the value remains because Streamlit is preserving the widget state.

The dimming effect

You mentioned:

the light dims

That usually happens when:

with st.spinner("Thinking..."):


is running.

Example:

with st.spinner("Searching knowledge base..."):
    answer = rag.answer_question(query)


During that period Streamlit greys out parts of the interface.

That's normal.

What I need to see

Show me the section of app.py containing:

st.chat_input(...)


or

st.text_input(...)


and the send button logic.

That's where the cause will be. Based on your symptoms, I'd bet there's a st.text_input whose session-state value isn't being cleared after submission.
