import pandas as pd

# File paths
input_csv = "static/leaderboard/leaderboard.csv"
output_html = "static/leaderboard/leaderboard.html"
page_title = "Leaderboard"

# Read CSV
df = pd.read_csv(input_csv)

# Ensure necessary columns exist
if "Name" not in df.columns or "Link" not in df.columns or "Type" not in df.columns:
    raise ValueError(f"CSV file {input_csv} must contain 'Name', 'Link', and 'Type' columns.")

# Add "Model" column before dropping `Name` and `Link`
df.insert(0, "Model", df.apply(lambda row: f'<a href="{row["Link"]}" target="_blank">{row["Name"]}</a>', axis=1))

# Drop unnecessary columns
df = df.drop(columns=["name (api)", "Link", "Name + citation"])  # Keeping relevant scores

# Define color coding for the "Type" column
type_colors = {
    "Large": "#ffebcc",      # Light orange
    "Medical": "#ccffcc",     # Light green
    "Small": "#cce5ff",      # Light blue
    "Reasoning": "#f5e6ff",  # Light purple
    "-": "#e6e6e6"           # Light gray for undefined types
}

def generate_row(row):
    row_color = type_colors.get(row["Type"], "#e6e6e6")  # Default to light gray
    row_html = f'<tr style="background-color: {row_color};">' + "".join(f"<td>{row[col]}</td>" for col in df.columns) + "</tr>"
    return row_html

# Generate HTML
df.sort_values(by="Overall", inplace=True)
table_headers = " ".join(f"<th>{col}</th>" for col in df.columns)
table_rows = " ".join(generate_row(row) for _, row in df.iterrows())

html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.min.css">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: white; }}
        table {{ width: 100%; border-collapse: collapse; background-color: white; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
        th {{ background-color: #f2f2f2; cursor: pointer; }}
        th:hover {{ background-color: #ddd; }}
        a {{ text-decoration: none; color: #007bff; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h4>{page_title}</h4>
    <table id="leaderboard" class="display">
        <thead>
            <tr>{table_headers}</tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
    <script>
        $(document).ready( function () {{
            $('#leaderboard').DataTable({{
                "order": [[ {df.columns.get_loc("Overall")}, "desc" ]], // Sort by "Overall" column descending
                "searching": false,
                "paging": false,
                "info": false,
                "columnDefs": [
                    {{ "type": "html", "targets": [0] }}
                ]
            }});
        }});
    </script>
</body>
</html>
"""

# Save to file
with open(output_html, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"Leaderboard generated: {output_html}")