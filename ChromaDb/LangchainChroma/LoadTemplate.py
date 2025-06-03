import json
import re

# Load the template from ChromaTemplate.json
with open("ChromaTemplate.json", "r") as file:
    templates = json.load(file)

# Find the RiskEventTemplate object
risk_event_template = None
for obj in templates:
    if "RiskEventTemplate" in obj:
        risk_event_template = obj["RiskEventTemplate"]
        break

# Load the audit data from auditData.json
with open("auditData.json", "r") as file:
    audit_data = json.load(file)

# Populate the template with values from auditData.json
populated_texts = []
pattern = re.compile(r"{(.*?)}")
for record in audit_data:
    keys = pattern.findall(risk_event_template)
    populated_text = risk_event_template
    for key in keys:
        populated_text = populated_text.replace(f"{{{key}}}", str(record.get(key, "")))
    # If ExtendedPropertiesAccess exists and is a dict, append its properties
    if "ExtendedPropertiesAccess" in record and isinstance(record["ExtendedPropertiesAccess"], dict):
        ext_props = record["ExtendedPropertiesAccess"]
        ext_props_str = " ".join(f"{k}: {v}" for k, v in ext_props.items())
        populated_text += " " + ext_props_str
    populated_texts.append(populated_text)

# Print the populated texts
for text in populated_texts:
    print(text)
