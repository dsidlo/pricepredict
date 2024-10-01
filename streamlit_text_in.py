import streamlit as st


class st_test_input(st):
    def __init__(self):
        self.st = st
        self.sdCounter = 0
        self.ti_value = ''
        self.txt_tiTextInputFld = f'tiTextInputFld-{self.sdCounter}'


    def clear_text_input():
        st.session_state['ti_value'] = ''


    def text_input_changed():
        st.session_state['ti_value'] = ''
        print(f'=== Text Input Changed [{st.session_state['sdCounter']}] [{st.session_state['ti_value']}] ===')

    def accept_text_input(st, txt_tiTextInputFld):
        print(f'--> accept_text_input: [{st.session_state['sdCounter']}] [{st.session_state['ti_value']}] ===')
        if bTestInput in st.session_state and st.session_state.bTestInput:
            # If-3
            if txt_tiTextInputFld in st.session_state:
                # If-4
                st.session_state[txt_tiTextInputFld] = st.empty()
                st.session_state['sdCounter'] += 1
                txt_tiTextInputFld = f'tiTextInputFld-{st.session_state['sdCounter']}'
        st.session_state['ti_entered'] = st.session_state['ti_value']
        print(f'<-- accept_text_input: [{st.session_state['sdCounter']}] [{st.session_state['ti_value']}] ===')


    if 'ti_value' not in st.session_state:
        st.session_state['ti_value'] = ''
    if 'sdCounter' not in st.session_state:
        st.session_state['sdCounter'] = 0

    if 'ti_value' in st.session_state:
        print(f'==[{st.session_state['sdCounter']}]== ti_value = {st.session_state['ti_value']}')


    if 'txt_tiTextInputFld' not in locals():
        # If-1
        txt_tiTextInputFld = f'tiTextInputFld-{st.session_state['sdCounter']}'
        print(f"if-1: not in locals() [{txt_tiTextInputFld}]")
    if ('txt_tiTextInputFld' not in st.session_state):
        # If-2
        txt_tiTextInputFld = f'tiTextInputFld-{st.session_state['sdCounter']}'
        st.session_state['txt_tiTextInputFld'] = txt_tiTextInputFld
        print(f" if-2: txt_tiTextInputFld not in locals() [{txt_tiTextInputFld}]")

    def clear_text_input(self):
        self.st.text_input('Text Area',  label_visibility='collapsed',
                            value=st.session_state['ti_value'], key=txt_tiTextInputFld,
                            on_change=text_input_changed)

st.title('Streamlit Test')


tiTextInputFld = st.text_input('Text Area',  label_visibility='collapsed',
                                value=st.session_state[f'ti_value'], key=txt_tiTextInputFld,
                                on_change=text_input_changed)

st.session_state['txt_tiTextInputFld'] = txt_tiTextInputFld

st.button('Clear Text Input', on_click=clear_text_input, key=f'bClearTex')
st.echo(f"ti_value = {st.session_state['ti_value']}")


