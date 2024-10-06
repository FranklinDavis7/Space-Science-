import pygal
import streamlit as st

# Create a line chart
line_chart = pygal.Line()

# Add a title to the chart
line_chart.title = 'Line Chart Example'

# Add data to the chart
line_chart.add('Series 1', [10, 20, 30, 40, 50], color='red')
line_chart.add('Series 2', [20, 30, 40, 50, 60], color='blue')

# Add labels to the chart
line_chart.x_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
line_chart.y_title = 'Values'

# Render the chart as an HTML string
chart_html = line_chart.render()

# Display the chart in Streamlit
st.components.v1.html(chart_html, width=800, height=600)