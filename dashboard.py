"""Entry point: streamlit run dashboard.py"""
import streamlit as st

from ui.theme import setup_page

setup_page("Smart EV Charging Scheduler")

nav = st.navigation([
    st.Page("views/simulate.py", title="Simulate a day", icon="🔋", default=True),
    st.Page("views/results.py", title="Results", icon="📊"),
    st.Page("views/how_it_works.py", title="How it works", icon="🧭"),
])
nav.run()
